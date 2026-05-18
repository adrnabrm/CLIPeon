# CLIPeon

Visual similarity search for Pokémon TCG card art using [OpenCLIP](https://github.com/mlfoundations/open_clip), [DINOv2](https://github.com/facebookresearch/dinov2), and [ChromaDB](https://www.trychroma.com/).

The pipeline downloads official card images, crops the artwork with layout-aware bounding boxes, embeds the crops using **three independent signals**, and supports nearest-neighbor search with late-fusion scoring against a persistent vector index.

## Current state

| Item | Status |
|------|--------|
| Sets in `data/raw` and `data/clean` | **me1**, **me2** |
| Clean art crops indexed | **318** cards (188 me1 + 130 me2) |
| Vector DB | Chroma at `data/chroma`, collections `cards_clip` / `cards_dino` / `cards_color` |
| CLIP model | ViT-B-32, OpenAI weights (512-d, cosine) |
| DINOv2 model | `dinov2_vitb14` via torch.hub (768-d, cosine) |
| Python env | Conda env `clipeon` (see [Setup](#setup)) |

Scripts are committed; `data/` is gitignored (regenerate locally).

## Pipeline

```
pokemon-tcg-data JSON
        │
        ▼
  gather_data.py          →  data/raw/<set>/<rarity>/<id>_large.png
        │
        ▼
  clean_data.py           →  data/clean/<set>/<id>.png
        │
        ▼
  clip_actions.py index   →  data/chroma/  (3 collections + index_params.json)
        │
        ▼
  clip_actions.py query   →  top-k similar cards (fused score + per-signal breakdown)
```

## Directory layout

```
CLIPeon/
├── gather_data.py      # Download raw card images from set JSON
├── clean_data.py       # Crop art from raw scans (layout detection)
├── clip_actions.py     # Three-signal embed, index, and similarity query
├── app.py              # Gradio web UI for visual search
├── requirements.txt    # Dependencies for clip_actions (clipeon env)
└── data/               # gitignored
    ├── raw/<set>/<rarity>/        # Full card PNGs (~733×1024)
    ├── clean/<set>/               # Cropped art only
    └── chroma/                    # Persistent vector store
        ├── cards_clip/            # CLIP embeddings (512-d)
        ├── cards_dino/            # DINOv2 embeddings (768-d)
        ├── cards_color/           # HSV color histogram embeddings (96-d)
        └── index_params.json      # Saved embedding parameters (auto-generated)
```

## Setup

### Conda environment

```bash
conda create -n clipeon python=3.10 -y
conda activate clipeon
pip install -r requirements.txt
```

`clean_data.py` and `gather_data.py` only need **Pillow** for cropping; the full stack (torch, open-clip, chromadb, opencv) is required for `clip_actions.py`.

DINOv2 weights are downloaded automatically via `torch.hub` on first indexing or query (~330 MB, cached to `~/.cache/torch/hub`). No extra pip install is needed.

### Card metadata (gather only)

`gather_data.py` reads set JSON from [pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data). It checks, in order:

1. `$POKEMON_TCG_CARDS_DIR/<set_id>.json`
2. `~/Projects/pokemon-tcg-data/en/cards/<set_id>.json`
3. `~/Projects/pokemon-tcg-data/cards/en/<set_id>.json`
4. `<repo>/../pokemon-tcg-data/...` (same paths)

## Scripts

### 1. `gather_data.py` — download raw images

Downloads `small` and/or `large` images from URLs in the set JSON into rarity subfolders.

```bash
python gather_data.py me1
python gather_data.py me2 --size large    # large only
python gather_data.py me1 -o /path/to/out
```

Output: `data/raw/<set_id>/<Rarity>/<card_id>_large.png`

### 2. `clean_data.py` — crop artwork

Walks all rarity folders under a set's raw directory, detects card layout per image, and writes flattened crops to `data/clean/<set_id>/`.

```bash
python clean_data.py me1
python clean_data.py me2 --force   # overwrite existing crops
```

#### Layout detection

Each full card (~733×1024) is classified into one of three layouts; the crop box is chosen automatically:

| Layout | Examples | Crop box `(left, top, right, bottom)` | Output size |
|--------|----------|----------------------------------------|-------------|
| **Standard** (silver art frame) | Commons, Illustration Rares | `(30, 110, 704, 510)` | 674×400 |
| **Full-art borderless** | Most SIRs, UR ex artwork | `(30, 30, 704, 870)` | 674×840 |
| **Full-art with header** | Wally's Compassion, some UR/SIR | `(30, 110, 704, 870)` | 674×760 |

Detection logic (see `clean_data.py`):

1. Silver border samples → standard layout  
2. Else bright row at **y=80** (title bar) → full-art with header  
3. Else → borderless full-art  

Shared helpers `crop_card_art()` and `is_full_card_image()` are imported by `clip_actions.py` for query-time cropping.

### 3. `clip_actions.py` — embed and search

#### How the three-signal embedding works

Each card is embedded three times — once per signal — and stored in three separate ChromaDB collections. Signals are kept independent so their weights can be tuned or changed without re-running the others.

**CLIP** (512-d) understands *what subject* is in the image. It knows a scene looks like an ocean, a forest, or a battle because it was trained on image–text pairs. It is good at matching cards with the same Pokémon or similar thematic content.

**DINOv2** (768-d) understands *how the image looks*. Trained purely on visual correspondence (no text), it captures texture, brushwork, composition, and painterly style. It is good at matching cards with a similar artistic style even if the subject differs — e.g., two different Pokémon both rendered in loose watercolor washes.

**HSV color histogram** (96-d) understands *what palette* is in the image. It splits pixels into 32 buckets per HSV channel (hue, saturation, value), giving 96 numbers that describe the color distribution. Hue is weighted heaviest by default because it identifies the actual color identity of a card (blue ocean, red fire, green forest).

#### How search works (late fusion)

At query time, the query image is embedded by all three models. Each collection independently returns its top-N most similar cards. The three ranked lists are then **union-merged** and each card receives a weighted final score:

```
final_score = clip_weight × clip_score + dino_weight × dino_score + color_weight × color_score
```

A card only needs to rank highly in **one or more** signals to be competitive. A card missed by CLIP but found by DINOv2 (similar style, different subject) can still surface in the top results. This is the key advantage over baking everything into a single hybrid vector: each axis remains independently tunable.

#### Tuning the embedding

All parameters are saved to `data/chroma/index_params.json` when you index. Query time (CLI and UI) automatically loads this file so index and query always stay in sync.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `--clip-weight` | `0.40` | How much CLIP's semantic (subject) signal contributes to the final score |
| `--dino-weight` | `0.40` | How much DINOv2's visual style signal contributes |
| `--color-weight` | `0.20` | How much the color palette contributes |
| `--color-bins` | `32` | Histogram resolution per HSV channel (more = finer color discrimination) |
| `--h-weight` | `2.0` | Relative importance of **hue** (what color) within the histogram |
| `--s-weight` | `1.0` | Relative importance of **saturation** (how vivid) |
| `--v-weight` | `0.5` | Relative importance of **value** (how bright/dark) |
| `--dino-model` | `dinov2_vitb14` | DINOv2 variant (`dinov2_vits14` for faster/smaller, `dinov2_vitl14` for stronger) |

**Any time you change these parameters, re-index with `--force`.**

The fusion weights (`clip_weight`, `dino_weight`, `color_weight`) are the exception — they are applied at query time and can be overridden per-query without re-indexing.

#### Index

Embeds every PNG under `data/clean/` into all three ChromaDB collections. DINOv2 weights are downloaded automatically on first run.

```bash
conda activate clipeon
python clip_actions.py index
python clip_actions.py index --force                        # drop and rebuild all three collections
python clip_actions.py index --force --color-weight 0.3    # adjust default fusion weight
python clip_actions.py index --force --h-weight 3.0        # hue even more dominant in palette signal
python clip_actions.py index --force --dino-model dinov2_vitl14  # use larger DINOv2
```

#### Query

Finds the **k** most similar indexed cards using three-signal late fusion. Query images are cropped with the same layout logic when they look like full card scans (width ≥ 700 and height ≥ 900); already-clean art is embedded as-is.

```bash
python clip_actions.py query path/to/card.png -k 5
python clip_actions.py query data/raw/me1/Special\ Illustration\ Rare/me1-181_large.png -k 3
python clip_actions.py query data/clean/me1/me1-181.png -k 5 --no-crop

# Override fusion weights at query time (no re-index needed)
python clip_actions.py query photo.jpg -k 5 --dino-weight 0.6 --clip-weight 0.2 --color-weight 0.2
```

Results print the **fused similarity score** and a per-signal breakdown so you can see which axis drove each match:

```
1. sv7-167  set=sv7  score=0.8731  (clip=0.812 dino=0.921 color=0.788)
   raw: data/raw/sv7/Special Illustration Rare/sv7-167_large.png
```

Programmatic API:

```python
from pathlib import Path
from clip_actions import query_similar

hits = query_similar(
    Path("photo.jpg"),
    k=5,
    db_path=Path("data/chroma"),
    raw_root=Path("data/raw"),
)
for hit in hits:
    print(hit.card_id, hit.score, hit.clip_score, hit.dino_score, hit.color_score)
```

### 4. `app.py` — web UI

A Gradio interface for visual search. Upload any card image or art crop and browse results in a gallery.

```bash
conda activate clipeon
python app.py
python app.py --port 7861 --host 0.0.0.0
```

The UI loads `data/chroma/index_params.json` at startup and uses the same embedding parameters that were used at index time. Both CLIP and DINOv2 models are loaded on startup; expect a brief delay before the first search.

Gallery captions show the fused score plus per-signal breakdown: `C:0.81 D:0.92 Col:0.79`.

Features:
- Full card scans are auto-cropped to artwork before searching
- **Top k** slider (1–12)
- **IR / SIR Only** filter (Illustration Rare / Special Illustration Rare)

#### Label eval tab

Use the **Label eval** tab to build a dataset for tuning fusion weights later:

1. A random card image is picked from `data/raw` (already-labeled queries are skipped). Enable **IR / SIR Only** to limit both random queries and search results to Illustration Rare / Special Illustration Rare.
2. Click **Run query** to fetch 12 similar cards.
3. Click a result, then rate it: **0** (not relevant), **1** (kinda relevant), **2** (ideal).
4. **Save & next** appends one JSON line to `data/eval/labels.jsonl` (only results you rated).
5. **Skip** or **New random** picks another query without saving.

```bash
python app.py --labels-path data/eval/labels.jsonl
```

Each saved record includes the query card id, paths, current `index_params` snapshot, and per-result ranks/scores/relevance. A future eval script can read this file to sweep `clip_weight` / `dino_weight` / `color_weight`.

## Typical workflow

```bash
# 1. Download (needs pokemon-tcg-data JSON)
python gather_data.py me1
python gather_data.py me2

# 2. Crop art
python clean_data.py me1
python clean_data.py me2

# 3. Index (downloads DINOv2 weights on first run) + search
conda activate clipeon
pip install -r requirements.txt
python clip_actions.py index --force
python clip_actions.py query data/raw/me2/Illustration\ Rare/me2-101_large.png -k 5

# 4. Or launch the web UI
python app.py
```

After adding or re-cropping cards, re-run `index` (use `--force` for a full rebuild, or rely on upsert for incremental updates to all three collections).

## Dependencies

| Package | Used by |
|---------|---------|
| Pillow | `clean_data.py`, `clip_actions.py`, `app.py` |
| torch, torchvision | `clip_actions.py` (CLIP + DINOv2 via torch.hub) |
| open-clip-torch | `clip_actions.py` (CLIP model) |
| chromadb | `clip_actions.py` (three vector collections) |
| opencv-python-headless | `clip_actions.py` (HSV color histograms) |
| numpy | `clip_actions.py` |
| gradio | `app.py` |

See `requirements.txt` for pinned minimum versions.
