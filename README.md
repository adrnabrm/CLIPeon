# CLIPeon

Visual similarity search for Pokémon TCG card art using [OpenCLIP](https://github.com/mlfoundations/open_clip) and [ChromaDB](https://www.trychroma.com/).

The pipeline downloads official card images, crops the artwork with layout-aware bounding boxes, embeds the crops with a hybrid CLIP + color signal, and supports nearest-neighbor search against a persistent vector index.

## Current state

| Item | Status |
|------|--------|
| Sets in `data/raw` and `data/clean` | **me1**, **me2** |
| Clean art crops indexed | **318** cards (188 me1 + 130 me2) |
| Vector DB | Chroma at `data/chroma`, collection `cards` |
| CLIP model | ViT-B-32, OpenAI weights (512-d, cosine) |
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
  clip_actions.py index   →  data/chroma/  (embeddings + metadata)
        │
        ▼
  clip_actions.py query   →  top-k similar cards (paths into raw)
```

## Directory layout

```
CLIPeon/
├── gather_data.py      # Download raw card images from set JSON
├── clean_data.py       # Crop art from raw scans (layout detection)
├── clip_actions.py     # CLIP embed, index, and similarity query
├── app.py              # Gradio web UI for visual search
├── requirements.txt    # Dependencies for clip_actions (clipeon env)
└── data/               # gitignored
    ├── raw/<set>/<rarity>/        # Full card PNGs (~733×1024)
    ├── clean/<set>/               # Cropped art only
    └── chroma/                    # Persistent vector store
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

#### How the embedding works

Each card is embedded as a **hybrid vector** that blends two signals:

**CLIP** understands *what* is in the image — whether a scene looks like an ocean, a forest, a battle. It's a 512-dimensional semantic embedding from OpenCLIP (ViT-B-32).

**HSV color histogram** understands *what colors* are in the image — how much red, blue, green, etc. is present, and how vivid or dark. It splits the image's colors into 32 buckets per HSV channel (hue, saturation, value), giving 96 numbers total.

The two are combined like this:

1. The color histogram is tiled to 512 dimensions to match CLIP, so both signals get equal dimensional representation.
2. Each is independently normalized to unit length, so neither dominates through scale.
3. They are weighted and added together: `(clip * clip_weight) + (color * color_weight)`.
4. The result is re-normalized to unit length and stored in ChromaDB.

This means `clip_weight=0.65, color_weight=0.2` genuinely means "65% CLIP, 20% color" — the weights aren't silently cancelled out by normalization.

#### Tuning the embedding

All parameters are saved to `data/chroma/index_params.json` when you index. Query time (CLI and UI) automatically loads this file so index and query always use the same values — no manual syncing needed.

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `--clip-weight` | `0.65` | How much CLIP's semantic understanding matters |
| `--color-weight` | `0.20` | How much the color palette matters |
| `--color-bins` | `32` | Histogram resolution per HSV channel (more bins = finer color discrimination) |
| `--h-weight` | `2.0` | Relative importance of **hue** (what color) within the histogram |
| `--s-weight` | `1.0` | Relative importance of **saturation** (how vivid) |
| `--v-weight` | `0.5` | Relative importance of **value** (how bright/dark) |

Hue is weighted heaviest by default because it captures the actual color identity of a card (blue ocean, red fire, green forest) and is the most useful signal for "find cards with a similar palette."

**Any time you change these parameters, re-index with `--force`.**

#### Index

Embeds every PNG under `data/clean/` and upserts into ChromaDB.

```bash
conda activate clipeon
python clip_actions.py index
python clip_actions.py index --force                        # drop and rebuild collection
python clip_actions.py index --force --color-weight 0.3    # more color influence
python clip_actions.py index --force --h-weight 3.0        # hue even more dominant
```

#### Query

Finds the **k** most similar indexed cards. Query images are cropped with the same layout logic when they look like full card scans (width ≥ 700 and height ≥ 900); already-clean art is embedded as-is.

```bash
python clip_actions.py query path/to/card.png -k 5
python clip_actions.py query data/raw/me1/Special\ Illustration\ Rare/me1-181_large.png -k 3
python clip_actions.py query data/clean/me1/me1-181.png -k 5 --no-crop
```

Results print **similarity** (`1 - cosine distance`) and the matching **raw** `*_large.png` path under `data/raw/`.

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
    print(hit.card_id, hit.score, hit.raw_path)
```

### 4. `app.py` — web UI

A Gradio interface for visual search. Upload any card image or art crop and browse results in a gallery.

```bash
conda activate clipeon
python app.py
python app.py --port 7861 --host 0.0.0.0
```

The UI loads `data/chroma/index_params.json` at startup and uses the same embedding parameters that were used at index time automatically.

Features:
- Full card scans are auto-cropped to artwork before searching
- **Top k** slider (1–20)
- **IR / SIR Only** filter (Illustration Rare / Special Illustration Rare)

## Typical workflow

```bash
# 1. Download (needs pokemon-tcg-data JSON)
python gather_data.py me1
python gather_data.py me2

# 2. Crop art
python clean_data.py me1
python clean_data.py me2

# 3. Index + search
conda activate clipeon
pip install -r requirements.txt
python clip_actions.py index --force
python clip_actions.py query data/raw/me2/Illustration\ Rare/me2-101_large.png -k 5

# 4. Or launch the web UI
python app.py
```

After adding or re-cropping cards, re-run `index` (use `--force` for a full rebuild, or rely on upsert for incremental updates).

## Dependencies

| Package | Used by |
|---------|---------|
| Pillow | `clean_data.py`, `clip_actions.py`, `app.py` |
| torch, torchvision, open-clip-torch | `clip_actions.py` |
| chromadb | `clip_actions.py` |
| opencv-python-headless | `clip_actions.py` (HSV color histograms) |
| numpy | `clip_actions.py` |
| gradio | `app.py` |

See `requirements.txt` for pinned minimum versions.
