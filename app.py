#!/usr/bin/env python3
"""
CLIPeon – Gradio card art search UI (three-signal fusion).

Usage (conda env clipeon):
  python app.py
  python app.py --port 7860 --db-path data/chroma --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import gradio as gr
    from PIL import Image

    from clip_actions import (
        CLIP_COLLECTION,
        COLOR_COLLECTION,
        DINO_COLLECTION,
        DEFAULT_DINO_MODEL,
        DEFAULT_MODEL,
        DEFAULT_PRETRAINED,
        SimilarCard,
        _fuse_scores,
        embed_color_vector,
        embed_image_clip,
        embed_image_dino,
        find_raw_image,
        load_clip,
        load_dino,
        load_index_params,
        load_query_image,
        open_chroma_collection,
        resolve_device,
    )
except ImportError:
    print(
        "Missing dependencies. Activate clipeon and run:\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
LABEL_K = 12

_clip_model = None
_clip_preprocess = None
_dino_model = None
_dino_transform = None
_clip_collection = None
_dino_collection = None
_color_collection = None
_raw_root: Path = PROJECT_ROOT / "data" / "raw"
_db_path: Path = PROJECT_ROOT / "data" / "chroma"
_labels_path: Path = PROJECT_ROOT / "data" / "eval" / "labels.jsonl"
_device = None
_index_params: dict = {}

FULL_ART_RARITIES = frozenset({
    "Illustration Rare",
    "Special Illustration Rare",
    "Trainer Gallery Rare Holo",
    "Rare Rainbow",
    "Rare Secret",
    "Rare Ultra",
})


def _ensure_loaded(
    model_name: str,
    pretrained: str,
    dino_model_name: str,
    db_path: Path,
) -> None:
    global _clip_model, _clip_preprocess, _dino_model, _dino_transform
    global _clip_collection, _dino_collection, _color_collection
    global _device, _index_params

    if _clip_model is None:
        _device = resolve_device(None)
        print(f"Loading CLIP {model_name}/{pretrained}...")
        _clip_model, _clip_preprocess = load_clip(model_name, pretrained, _device)
        print(f"Loading DINOv2 {dino_model_name}...")
        _dino_model, _dino_transform = load_dino(dino_model_name, _device)

    if _clip_collection is None:
        _clip_collection = open_chroma_collection(db_path, CLIP_COLLECTION)
        _dino_collection = open_chroma_collection(db_path, DINO_COLLECTION)
        _color_collection = open_chroma_collection(db_path, COLOR_COLLECTION)
        _index_params = load_index_params(db_path)


def _raw_path_is_full_art(raw_path: Path) -> bool:
    return bool(set(raw_path.parts) & FULL_ART_RARITIES)


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_project_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def _discover_raw_cards(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        return []
    return sorted(raw_root.glob("**/*_large.png"))


def _parse_raw_card(path: Path, raw_root: Path) -> tuple[str, str]:
    card_id = path.stem.removesuffix("_large")
    set_id = path.resolve().relative_to(raw_root.resolve()).parts[0]
    return card_id, set_id


def _load_labeled_query_paths(labels_path: Path) -> set[str]:
    if not labels_path.is_file():
        return set()
    paths: set[str] = set()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw = record.get("query_raw_path")
        if raw:
            paths.add(raw)
            paths.add(str(_resolve_project_path(raw)))
    return paths


def _pick_random_raw(exclude: set[str], *, full_art_only: bool = False) -> Path | None:
    pool = _discover_raw_cards(_raw_root)
    available = [
        p for p in pool
        if _relative_path(p) not in exclude and str(p.resolve()) not in exclude
        and (not full_art_only or _raw_path_is_full_art(p))
    ]
    if not available:
        return None
    return random.choice(available)


def _hit_to_dict(hit: SimilarCard) -> dict:
    return {
        "card_id": hit.card_id,
        "set_id": hit.set_id,
        "score": hit.score,
        "clip_score": hit.clip_score,
        "dino_score": hit.dino_score,
        "color_score": hit.color_score,
        "clean_path": hit.clean_path,
    }


def _dict_to_hit(data: dict) -> SimilarCard:
    card_id = data["card_id"]
    set_id = data["set_id"]
    return SimilarCard(
        card_id=card_id,
        set_id=set_id,
        score=float(data["score"]),
        clip_score=float(data["clip_score"]),
        dino_score=float(data["dino_score"]),
        color_score=float(data["color_score"]),
        raw_path=find_raw_image(card_id, set_id, _raw_root),
        clean_path=data.get("clean_path"),
    )


def _search_hits(
    query_image: Image.Image,
    k: int,
    *,
    full_art_only: bool = False,
) -> list[SimilarCard]:
    dino_model_name = _index_params.get("dino_model", DEFAULT_DINO_MODEL)
    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, dino_model_name, _db_path)

    params = _index_params
    q_clip = embed_image_clip(_clip_model, _clip_preprocess, query_image, _device)
    q_dino = embed_image_dino(_dino_model, _dino_transform, query_image, _device)
    q_color = embed_color_vector(
        query_image,
        params["color_bins"],
        params["h_weight"],
        params["s_weight"],
        params["v_weight"],
    )

    base_k = int(k)
    min_pool = 400 if full_art_only else 150
    candidate_n = min(max(base_k * 3, min_pool), _clip_collection.count())

    clip_results = _clip_collection.query(
        query_embeddings=[q_clip],
        n_results=candidate_n,
        include=["metadatas", "distances"],
    )
    dino_results = _dino_collection.query(
        query_embeddings=[q_dino],
        n_results=candidate_n,
        include=["distances"],
    )
    color_results = _color_collection.query(
        query_embeddings=[q_color],
        n_results=candidate_n,
        include=["distances"],
    )

    fused = _fuse_scores(
        clip_results,
        dino_results,
        color_results,
        params["clip_weight"],
        params["dino_weight"],
        params["color_weight"],
        clip_col=_clip_collection,
        dino_col=_dino_collection,
        color_col=_color_collection,
        q_clip=q_clip,
        q_dino=q_dino,
        q_color=q_color,
    )

    meta_lookup: dict[str, dict] = {
        id_: meta
        for id_, meta in zip(clip_results["ids"][0], clip_results["metadatas"][0])
    }
    missing = [t[0] for t in fused[: base_k * 2] if t[0] not in meta_lookup]
    if missing:
        extra = _clip_collection.get(ids=missing, include=["metadatas"])
        for id_, meta in zip(extra["ids"], extra["metadatas"]):
            meta_lookup[id_] = meta

    max_fused = params["clip_weight"] + params["dino_weight"] + params["color_weight"]
    self_match_threshold = max_fused * 0.98

    hits: list[SimilarCard] = []
    for card_id, fused_score, cs, ds, co in fused:
        if len(hits) >= base_k:
            break
        if fused_score >= self_match_threshold:
            continue

        meta = meta_lookup.get(card_id, {})
        set_id = meta.get("set_id", "")
        raw_path = find_raw_image(card_id, set_id, _raw_root)

        if full_art_only:
            if raw_path is None or not _raw_path_is_full_art(raw_path):
                continue

        hits.append(
            SimilarCard(
                card_id=card_id,
                set_id=set_id,
                score=fused_score,
                clip_score=cs,
                dino_score=ds,
                color_score=co,
                raw_path=raw_path,
                clean_path=meta.get("path"),
            )
        )
    return hits


def _hits_to_gallery(
    hits: list[SimilarCard],
    label_map: dict[str, int] | None = None,
) -> list[tuple[Image.Image, str]]:
    gallery: list[tuple[Image.Image, str]] = []
    for hit in hits:
        img: Image.Image | None = None
        if hit.raw_path and hit.raw_path.is_file():
            img = Image.open(hit.raw_path).convert("RGB")
        elif hit.clean_path:
            clean_abs = PROJECT_ROOT / hit.clean_path
            if clean_abs.is_file():
                img = Image.open(clean_abs).convert("RGB")

        if img is None:
            continue

        prefix = ""
        if label_map is not None and hit.card_id in label_map:
            prefix = f"[{label_map[hit.card_id]}] "
        caption = (
            f"{prefix}{hit.card_id}  ·  {hit.set_id}  ·  {hit.score:.4f}"
            f"  (C:{hit.clip_score:.2f} D:{hit.dino_score:.2f} Col:{hit.color_score:.2f})"
        )
        gallery.append((img, caption))
    return gallery


def search(
    query_image: Image.Image | None,
    k: int,
    full_art_only: bool = False,
) -> list[tuple[Image.Image, str]]:
    if query_image is None:
        return []

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        query_image.save(tmp_path)

    try:
        loaded = load_query_image(tmp_path, crop=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    hits = _search_hits(loaded, int(k), full_art_only=full_art_only)
    return _hits_to_gallery(hits)


def _label_status_text(
    query_path: str | None,
    label_map: dict[str, int],
    *,
    hits_count: int = 0,
    selected_id: str | None = None,
) -> str:
    if not query_path:
        return "No query card loaded. Click **New random** or **Skip**."
    path = _resolve_project_path(query_path)
    card_id, set_id = _parse_raw_card(path, _raw_root)
    lines = [
        f"**Query:** `{card_id}` · set `{set_id}`",
        f"Labeled **{len(label_map)}** result(s)",
    ]
    if hits_count:
        lines.append(f"Showing **{hits_count}** results (run query if empty)")
    if selected_id:
        lines.append(f"Selected: `{selected_id}`")
    lines.append("")
    lines.append("Select a result in the gallery, then click **0**, **1**, or **2**.")
    lines.append("- **0** = not relevant · **1** = kinda relevant · **2** = ideal")
    return "\n\n".join(lines)


def label_new_random(full_art_only: bool = False) -> tuple:
    exclude = _load_labeled_query_paths(_labels_path)
    path = _pick_random_raw(exclude, full_art_only=full_art_only)
    if path is None:
        if full_art_only:
            gr.Warning(
                "No unlabeled full art cards available. Try turning off Full Art Only."
            )
        else:
            gr.Warning(
                "No unlabeled raw cards available (all may already be in labels.jsonl)."
            )
        return (
            None,
            "",
            [],
            [],
            {},
            None,
            _label_status_text(None, {}),
        )

    img = Image.open(path).convert("RGB")
    rel = _relative_path(path)
    card_id, set_id = _parse_raw_card(path, _raw_root)
    status = (
        f"**Query:** `{card_id}` · set `{set_id}`\n\n"
        "Click **Run query** to fetch 12 similar cards, then rate results."
    )
    return (img, rel, [], [], {}, None, status)


def label_run_query(query_path: str, full_art_only: bool = False) -> tuple:
    if not query_path:
        gr.Warning("No query card loaded.")
        return [], [], {}, None, _label_status_text(None, {})

    path = _resolve_project_path(query_path)
    if not path.is_file():
        gr.Warning(f"Query image not found: {path}")
        return [], [], {}, None, _label_status_text(query_path, {})

    loaded = load_query_image(path, crop=True)
    hits = _search_hits(loaded, LABEL_K, full_art_only=full_art_only)
    hit_dicts = [_hit_to_dict(h) for h in hits]
    gallery = _hits_to_gallery(hits, {})
    status = _label_status_text(query_path, {}, hits_count=len(hits))
    return gallery, hit_dicts, {}, None, status


def label_gallery_select(
    evt: gr.SelectData,
    hit_dicts: list[dict],
) -> str | None:
    if evt is None or evt.index is None or not hit_dicts:
        return None
    idx = evt.index
    if idx < 0 or idx >= len(hit_dicts):
        return None
    return hit_dicts[idx]["card_id"]


def label_rate(
    relevance: int,
    query_path: str,
    hit_dicts: list[dict],
    label_map: dict[str, int],
    selected_id: str | None,
) -> tuple:
    if not selected_id:
        gr.Warning("Select a result in the gallery first.")
        hits = [_dict_to_hit(d) for d in hit_dicts]
        return label_map, _hits_to_gallery(hits, label_map), _label_status_text(
            query_path, label_map, hits_count=len(hits), selected_id=selected_id
        )

    updated = dict(label_map)
    updated[selected_id] = relevance
    hits = [_dict_to_hit(d) for d in hit_dicts]
    gallery = _hits_to_gallery(hits, updated)
    status = _label_status_text(
        query_path, updated, hits_count=len(hits), selected_id=selected_id
    )
    return updated, gallery, status


def _append_label_record(
    query_path: str,
    hit_dicts: list[dict],
    label_map: dict[str, int],
    *,
    full_art_only: bool = False,
) -> None:
    path = _resolve_project_path(query_path)
    card_id, set_id = _parse_raw_card(path, _raw_root)
    hits = [_dict_to_hit(d) for d in hit_dicts]

    results: list[dict] = []
    for rank, hit in enumerate(hits, start=1):
        results.append({
            "card_id": hit.card_id,
            "set_id": hit.set_id,
            "rank": rank,
            "relevance": label_map.get(hit.card_id, 0),
            "fused_score": round(hit.score, 4),
            "clip_score": round(hit.clip_score, 3),
            "dino_score": round(hit.dino_score, 3),
            "color_score": round(hit.color_score, 3),
        })

    record = {
        "query_card_id": card_id,
        "query_set_id": set_id,
        "query_raw_path": _relative_path(path),
        "labeled_at": datetime.now(timezone.utc).isoformat(),
        "k": LABEL_K,
        "full_art_only": full_art_only,
        "index_params": dict(_index_params),
        "results": results,
    }

    _labels_path.parent.mkdir(parents=True, exist_ok=True)
    with _labels_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def label_save_and_next(
    query_path: str,
    hit_dicts: list[dict],
    label_map: dict[str, int],
    full_art_only: bool = False,
) -> tuple:
    if not any(r > 0 for r in label_map.values()):
        gr.Warning("Rate at least one result as 1 or 2 before saving.")
        hits = [_dict_to_hit(d) for d in hit_dicts]
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            _label_status_text(query_path, label_map, hits_count=len(hits)),
        )

    if not query_path or not hit_dicts:
        gr.Warning("Run query before saving.")
        return label_new_random(full_art_only)

    _append_label_record(
        query_path, hit_dicts, label_map, full_art_only=full_art_only
    )
    gr.Info(f"Saved labels to {_relative_path(_labels_path)}")
    return label_new_random(full_art_only)


def label_skip(full_art_only: bool = False) -> tuple:
    return label_new_random(full_art_only)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CLIPeon – Card Art Search") as demo:
        gr.Markdown("# CLIPeon")
        gr.Markdown(
            "Pokémon TCG card art search with three-signal fusion: "
            "**CLIP** (subject) · **DINOv2** (style) · **Color** (palette)."
        )

        with gr.Tabs():
            with gr.Tab("Search"):
                gr.Markdown(
                    "Upload any card image or art crop. Full card scans are "
                    "auto-cropped to artwork before searching."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Query image",
                            height=320,
                        )
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=12,
                            step=1,
                            value=5,
                            label="Top k results",
                        )
                        full_art_checkbox = gr.Checkbox(
                            label="Full Art Only (IR / SIR / Rare Ultra / TG / GG)",
                            value=False,
                        )
                        search_btn = gr.Button("Search", variant="primary")

                results_gallery = gr.Gallery(
                    label="Similar cards",
                    columns=5,
                    object_fit="contain",
                    height="auto",
                    show_label=True,
                )

                search_btn.click(
                    fn=search,
                    inputs=[image_input, k_slider, full_art_checkbox],
                    outputs=results_gallery,
                )

            with gr.Tab("Label eval"):
                gr.Markdown(
                    "Build a labeled dataset for future weight tuning. "
                    "A random raw card is shown as the query; run search, rate "
                    "results (**0** = not relevant, **1** = kinda, **2** = ideal), "
                    "then **Save & next**. Labels append to "
                    f"`{_relative_path(_labels_path)}`."
                )

                label_query_path = gr.State("")
                label_hit_dicts = gr.State([])
                label_map = gr.State({})
                label_selected = gr.State(None)

                with gr.Row():
                    with gr.Column(scale=1):
                        label_query_image = gr.Image(
                            type="pil",
                            label="Query card (random from raw)",
                            height=320,
                            interactive=False,
                        )
                        label_full_art_checkbox = gr.Checkbox(
                            value=False,
                            label="Full Art Only (IR / SIR / Rare Ultra / TG / GG)",
                        )
                        with gr.Row():
                            label_run_btn = gr.Button("Run query", variant="primary")
                            label_random_btn = gr.Button("New random")
                            label_skip_btn = gr.Button("Skip")
                        label_save_btn = gr.Button("Save & next", variant="primary")
                        label_status = gr.Markdown()

                    with gr.Column(scale=2):
                        label_results_gallery = gr.Gallery(
                            label="Results (select one, then rate)",
                            columns=4,
                            object_fit="contain",
                            height="auto",
                            show_label=True,
                        )
                        with gr.Row():
                            label_btn_0 = gr.Button("0 — Not relevant")
                            label_btn_1 = gr.Button("1 — Kinda relevant")
                            label_btn_2 = gr.Button("2 — Ideal")

                label_outputs = [
                    label_query_image,
                    label_query_path,
                    label_results_gallery,
                    label_hit_dicts,
                    label_map,
                    label_selected,
                    label_status,
                ]

                demo.load(
                    fn=label_new_random,
                    inputs=[label_full_art_checkbox],
                    outputs=label_outputs,
                )

                label_random_btn.click(
                    fn=label_new_random,
                    inputs=[label_full_art_checkbox],
                    outputs=label_outputs,
                )
                label_skip_btn.click(
                    fn=label_skip,
                    inputs=[label_full_art_checkbox],
                    outputs=label_outputs,
                )

                label_run_btn.click(
                    fn=label_run_query,
                    inputs=[label_query_path, label_full_art_checkbox],
                    outputs=[
                        label_results_gallery,
                        label_hit_dicts,
                        label_map,
                        label_selected,
                        label_status,
                    ],
                )

                label_results_gallery.select(
                    fn=label_gallery_select,
                    inputs=[label_hit_dicts],
                    outputs=label_selected,
                ).then(
                    fn=lambda qp, hd, lm, sid: _label_status_text(
                        qp, lm, hits_count=len(hd), selected_id=sid
                    ),
                    inputs=[label_query_path, label_hit_dicts, label_map, label_selected],
                    outputs=label_status,
                )

                label_btn_0.click(
                    fn=lambda qp, hd, lm, sid: label_rate(0, qp, hd, lm, sid),
                    inputs=[label_query_path, label_hit_dicts, label_map, label_selected],
                    outputs=[label_map, label_results_gallery, label_status],
                )
                label_btn_1.click(
                    fn=lambda qp, hd, lm, sid: label_rate(1, qp, hd, lm, sid),
                    inputs=[label_query_path, label_hit_dicts, label_map, label_selected],
                    outputs=[label_map, label_results_gallery, label_status],
                )
                label_btn_2.click(
                    fn=lambda qp, hd, lm, sid: label_rate(2, qp, hd, lm, sid),
                    inputs=[label_query_path, label_hit_dicts, label_map, label_selected],
                    outputs=[label_map, label_results_gallery, label_status],
                )

                label_save_btn.click(
                    fn=label_save_and_next,
                    inputs=[
                        label_query_path,
                        label_hit_dicts,
                        label_map,
                        label_full_art_checkbox,
                    ],
                    outputs=label_outputs,
                )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIPeon Gradio search UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "chroma")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "labels.jsonl",
        help="JSONL file for eval labels (default: data/eval/labels.jsonl)",
    )
    args = parser.parse_args()

    global _raw_root, _db_path, _labels_path
    _raw_root = args.raw_dir.expanduser().resolve()
    _db_path = args.db_path.expanduser().resolve()
    _labels_path = args.labels_path.expanduser().resolve()

    if not args.db_path.is_dir():
        print(
            f"Chroma DB not found at {args.db_path}. Run:\n"
            "  python clip_actions.py index",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not _raw_root.is_dir():
        print(
            f"Raw directory not found at {_raw_root}. Run gather_data.py first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("Pre-loading models and collections...")
    saved = load_index_params(_db_path)
    dino_model_name = saved.get("dino_model", DEFAULT_DINO_MODEL)
    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, dino_model_name, _db_path)
    print(f"Collections ready: {_clip_collection.count()} cards indexed.")

    raw_count = len(_discover_raw_cards(_raw_root))
    labeled_count = len(_load_labeled_query_paths(_labels_path))
    print(f"Raw cards: {raw_count}  |  Already labeled queries: {labeled_count}")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
