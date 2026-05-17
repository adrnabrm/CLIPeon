#!/usr/bin/env python3
"""
CLIPeon – Gradio card art search UI (three-signal fusion).

Usage (conda env clipeon):
  python app.py
  python app.py --port 7860 --db-path data/chroma --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

try:
    import gradio as gr
    from PIL import Image

    from clip_actions import (
        CLIP_COLLECTION,
        DINO_COLLECTION,
        COLOR_COLLECTION,
        DEFAULT_MODEL,
        DEFAULT_PRETRAINED,
        DEFAULT_DINO_MODEL,
        SimilarCard,
        embed_image_clip,
        embed_image_dino,
        embed_color_vector,
        load_clip,
        load_dino,
        load_index_params,
        load_query_image,
        open_chroma_collection,
        find_raw_image,
        resolve_device,
        _fuse_scores,
    )
except ImportError:
    print(
        "Missing dependencies. Activate clipeon and run:\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)

PROJECT_ROOT = Path(__file__).resolve().parent

_clip_model = None
_clip_preprocess = None
_dino_model = None
_dino_transform = None
_clip_collection = None
_dino_collection = None
_color_collection = None
_raw_root: Path = PROJECT_ROOT / "data" / "raw"
_device = None
_index_params: dict = {}

FULL_ART_RARITIES = frozenset({
    "Illustration Rare",
    "Special Illustration Rare",
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


def search(
    query_image: Image.Image | None,
    k: int,
    full_art_only: bool = False,
) -> list[tuple[Image.Image, str]]:
    if query_image is None:
        return []

    dino_model_name = _index_params.get("dino_model", DEFAULT_DINO_MODEL)
    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, dino_model_name,
                   PROJECT_ROOT / "data" / "chroma")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        query_image.save(tmp_path)

    try:
        loaded = load_query_image(tmp_path, crop=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    params = _index_params
    q_clip = embed_image_clip(_clip_model, _clip_preprocess, loaded, _device)
    q_dino = embed_image_dino(_dino_model, _dino_transform, loaded, _device)
    q_color = embed_color_vector(
        loaded,
        params["color_bins"],
        params["h_weight"],
        params["s_weight"],
        params["v_weight"],
    )

    # Over-fetch when filtering for full art so we have enough candidates after
    # the rarity filter.  Also over-fetch for late fusion so each signal can
    # surface cards the others might have ranked lower.
    base_k = int(k)
    multiplier = 20 if full_art_only else 5
    candidate_n = min(base_k * multiplier, _clip_collection.count())

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
        clip_results, dino_results, color_results,
        params["clip_weight"],
        params["dino_weight"],
        params["color_weight"],
    )

    # Build a metadata lookup from the clip collection results
    meta_lookup: dict[str, dict] = {
        id_: meta
        for id_, meta in zip(clip_results["ids"][0], clip_results["metadatas"][0])
    }
    # Back-fill metadata for any card_id that wasn't in the clip top-N but made
    # the cut after fusion (rare edge case with very unequal signal scores)
    missing = [t[0] for t in fused[:base_k * 2] if t[0] not in meta_lookup]
    if missing:
        extra = _clip_collection.get(ids=missing, include=["metadatas"])
        for id_, meta in zip(extra["ids"], extra["metadatas"]):
            meta_lookup[id_] = meta

    # Max possible fused score when all signals return score=1 (exact self-match)
    max_fused = params["clip_weight"] + params["dino_weight"] + params["color_weight"]
    self_match_threshold = max_fused * 0.98

    gallery: list[tuple[Image.Image, str]] = []
    for card_id, fused_score, cs, ds, co in fused:
        if len(gallery) >= base_k:
            break

        # Skip exact self-matches (query image is in the index)
        if fused_score >= self_match_threshold:
            continue

        meta = meta_lookup.get(card_id, {})
        set_id = meta.get("set_id", "")
        raw_path = find_raw_image(card_id, set_id, _raw_root)

        if full_art_only:
            if raw_path is None or not _raw_path_is_full_art(raw_path):
                continue

        img: Image.Image | None = None
        if raw_path and raw_path.is_file():
            img = Image.open(raw_path).convert("RGB")

        if img is None:
            clean_rel = meta.get("path")
            if clean_rel:
                clean_abs = PROJECT_ROOT / clean_rel
                if clean_abs.is_file():
                    img = Image.open(clean_abs).convert("RGB")

        if img is not None:
            caption = (
                f"{card_id}  ·  {set_id}  ·  {fused_score:.4f}"
                f"  (C:{cs:.2f} D:{ds:.2f} Col:{co:.2f})"
            )
            gallery.append((img, caption))

    return gallery


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CLIPeon – Card Art Search") as demo:
        gr.Markdown("## CLIPeon – Card Art Search")
        gr.Markdown(
            "Upload any Pokémon TCG card image or art crop. "
            "Full card scans are automatically cropped to their artwork before searching. "
            "Results are ranked by three-signal fusion: **CLIP** (subject) · "
            "**DINOv2** (style) · **Color** (palette)."
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
                    maximum=20,
                    step=1,
                    value=5,
                    label="Top k results",
                )
                full_art_checkbox = gr.Checkbox(
                    value=False,
                    label="IR / SIR Only (Illustration Rare / Special Illustration Rare)",
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

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIPeon Gradio search UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--db-path", type=Path, default=PROJECT_ROOT / "data" / "chroma")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    global _raw_root
    _raw_root = args.raw_dir.expanduser().resolve()

    if not args.db_path.is_dir():
        print(
            f"Chroma DB not found at {args.db_path}. Run:\n"
            "  python clip_actions.py index",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("Pre-loading models and collections...")
    saved = load_index_params(args.db_path)
    dino_model_name = saved.get("dino_model", DEFAULT_DINO_MODEL)
    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, dino_model_name, args.db_path)
    print(f"Collections ready: {_clip_collection.count()} cards indexed.")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
