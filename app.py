#!/usr/bin/env python3
"""
CLIPeon – Gradio card art search UI.

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
        DEFAULT_COLLECTION,
        DEFAULT_MODEL,
        DEFAULT_PRETRAINED,
        SimilarCard,
        embed_image,
        load_clip,
        load_query_image,
        open_chroma_collection,
        find_raw_image,
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

_model = None
_preprocess = None
_collection = None
_raw_root: Path = PROJECT_ROOT / "data" / "raw"
_device = None


def _ensure_loaded(model_name: str, pretrained: str, db_path: Path, collection_name: str) -> None:
    global _model, _preprocess, _collection, _device
    if _model is None:
        _device = resolve_device(None)
        _model, _preprocess = load_clip(model_name, pretrained, _device)
    if _collection is None:
        _collection = open_chroma_collection(db_path, collection_name)


def search(query_image: Image.Image | None, k: int) -> list[tuple[Image.Image, str]]:
    if query_image is None:
        return []

    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, PROJECT_ROOT / "data" / "chroma", DEFAULT_COLLECTION)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        query_image.save(tmp_path)

    try:
        loaded = load_query_image(tmp_path, crop=True)
    finally:
        tmp_path.unlink(missing_ok=True)

    embedding = embed_image(_model, _preprocess, loaded, _device)

    results = _collection.query(
        query_embeddings=[embedding],
        n_results=min(int(k), _collection.count()),
        include=["metadatas", "distances"],
    )

    gallery: list[tuple[Image.Image, str]] = []
    for card_id, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist
        if score >= 0.9999:
            continue
        set_id = meta["set_id"]
        caption = f"{card_id}  ·  {set_id}  ·  {score:.4f}"

        img: Image.Image | None = None

        raw_path = find_raw_image(card_id, set_id, _raw_root)
        if raw_path and raw_path.is_file():
            img = Image.open(raw_path).convert("RGB")

        if img is None:
            clean_rel = meta.get("path")
            if clean_rel:
                clean_abs = PROJECT_ROOT / clean_rel
                if clean_abs.is_file():
                    img = Image.open(clean_abs).convert("RGB")

        if img is not None:
            gallery.append((img, caption))

    return gallery


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="CLIPeon – Card Art Search") as demo:
        gr.Markdown("## CLIPeon – Card Art Search")
        gr.Markdown(
            "Upload any Pokémon TCG card image or art crop. "
            "Full card scans are automatically cropped to their artwork before searching."
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
            inputs=[image_input, k_slider],
            outputs=results_gallery,
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="CLIPeon Gradio search UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "chroma",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    args = parser.parse_args()

    global _raw_root
    _raw_root = args.raw_dir.expanduser().resolve()

    if not args.db_path.is_dir():
        print(f"Chroma DB not found at {args.db_path}. Run: python clip_actions.py index", file=sys.stderr)
        raise SystemExit(1)

    print(f"Pre-loading CLIP model and collection...")
    _ensure_loaded(DEFAULT_MODEL, DEFAULT_PRETRAINED, args.db_path, DEFAULT_COLLECTION)
    print(f"Collection has {_collection.count()} cards indexed.")

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
