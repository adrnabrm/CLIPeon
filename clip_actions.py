#!/usr/bin/env python3
"""
CLIP indexing and search for cleaned Pokémon TCG card art.

Usage (conda env clipeon):
  python clip_actions.py index
  python clip_actions.py index --force
  python clip_actions.py query path/to/image.png -k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import chromadb
    import numpy as np
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from clean_data import crop_card_art, is_full_card_image
except ImportError:
    print(
        "Missing dependencies. Activate clipeon and run:\n"
        "  pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1)

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
DEFAULT_COLLECTION = "cards"
EMBED_BATCH_SIZE = 32
DEFAULT_COLOR_BINS = 32
DEFAULT_CLIP_WEIGHT = 0.65
DEFAULT_COLOR_WEIGHT = 0.35


@dataclass(frozen=True)
class CardImage:
    card_id: str
    set_id: str
    path: Path


@dataclass(frozen=True)
class SimilarCard:
    card_id: str
    set_id: str
    score: float
    raw_path: Path | None
    clean_path: str | None


def color_histogram(pil_img: Image.Image, bins: int = DEFAULT_COLOR_BINS) -> np.ndarray:
    """Compute a normalized HSV color histogram from a PIL RGB image.

    Returns a float32 array of shape (3*bins,): H bins (0-180) followed by
    S and V bins (0-256), all concatenated and L1-normalized so they sum to 1.
    """
    hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    return hist / hist.sum()


def make_hybrid_vector(
    clip_vec: list[float] | np.ndarray,
    color_vec: np.ndarray,
    clip_weight: float = DEFAULT_CLIP_WEIGHT,
    color_weight: float = DEFAULT_COLOR_WEIGHT,
) -> list[float]:
    """Combine a CLIP embedding and a color histogram into one L2-normalized vector.

    The output dimension is len(clip_vec) + len(color_vec) (608 for ViT-B-32 + 3*32 bins).
    """
    cv = np.array(clip_vec, dtype=np.float32)
    combined = np.concatenate([cv * clip_weight, color_vec * color_weight])
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    return combined.tolist()


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


INDEX_PARAMS_FILE = "index_params.json"


def save_index_params(db_path: Path, clip_weight: float, color_weight: float, color_bins: int) -> None:
    params = {"clip_weight": clip_weight, "color_weight": color_weight, "color_bins": color_bins}
    (db_path / INDEX_PARAMS_FILE).write_text(json.dumps(params, indent=2))


def load_index_params(db_path: Path) -> dict:
    """Load saved index parameters, falling back to defaults if the file is missing."""
    params_path = db_path / INDEX_PARAMS_FILE
    if params_path.is_file():
        return json.loads(params_path.read_text())
    return {
        "clip_weight": DEFAULT_CLIP_WEIGHT,
        "color_weight": DEFAULT_COLOR_WEIGHT,
        "color_bins": DEFAULT_COLOR_BINS,
    }


def discover_cards(clean_root: Path) -> list[CardImage]:
    cards: list[CardImage] = []
    for set_dir in sorted(p for p in clean_root.iterdir() if p.is_dir()):
        for path in sorted(set_dir.glob("*.png")):
            cards.append(
                CardImage(card_id=path.stem, set_id=set_dir.name, path=path)
            )
    return cards


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(model_name: str, pretrained: str, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    model.to(device)
    return model, preprocess


class _ImageDataset(Dataset):
    def __init__(self, cards: list[CardImage], preprocess):
        self.cards = cards
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, idx: int):
        card = self.cards[idx]
        image = Image.open(card.path).convert("RGB")
        return self.preprocess(image), image, card


def embed_cards(
    model,
    preprocess,
    cards: list[CardImage],
    device: torch.device,
    batch_size: int,
    clip_weight: float = DEFAULT_CLIP_WEIGHT,
    color_weight: float = DEFAULT_COLOR_WEIGHT,
    color_bins: int = DEFAULT_COLOR_BINS,
) -> list[list[float]]:
    if not cards:
        return []

    dataset = _ImageDataset(cards, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch]),
            [b[1] for b in batch],   # PIL images for color histogram
            [b[2] for b in batch],   # CardImage metadata
        ),
    )

    vectors: list[list[float]] = []
    with torch.inference_mode():
        for images, pil_images, _batch_cards in loader:
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            clip_vecs = feats.cpu().tolist()
            for clip_vec, pil_img in zip(clip_vecs, pil_images):
                color_vec = color_histogram(pil_img, bins=color_bins)
                vectors.append(make_hybrid_vector(clip_vec, color_vec, clip_weight, color_weight))
    return vectors


def load_query_image(image_path: Path, *, crop: bool = True) -> Image.Image:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    if crop and is_full_card_image(rgb):
        cropped, _box = crop_card_art(rgb)
        return cropped
    return rgb


def embed_image(
    model,
    preprocess,
    image: Image.Image,
    device: torch.device,
) -> list[float]:
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feats = model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().tolist()[0]


def embed_image_hybrid(
    model,
    preprocess,
    image: Image.Image,
    device: torch.device,
    clip_weight: float = DEFAULT_CLIP_WEIGHT,
    color_weight: float = DEFAULT_COLOR_WEIGHT,
    color_bins: int = DEFAULT_COLOR_BINS,
) -> list[float]:
    """Produce a hybrid CLIP + HSV color histogram vector for a single image."""
    clip_vec = embed_image(model, preprocess, image, device)
    color_vec = color_histogram(image, bins=color_bins)
    return make_hybrid_vector(clip_vec, color_vec, clip_weight, color_weight)


def find_raw_image(card_id: str, set_id: str, raw_root: Path) -> Path | None:
    set_dir = raw_root / set_id
    if not set_dir.is_dir():
        return None
    matches = sorted(set_dir.glob(f"**/{card_id}_large.png"))
    return matches[0] if matches else None


def open_chroma_collection(db_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(name=collection_name)


def query_similar(
    image_path: Path,
    k: int,
    *,
    db_path: Path,
    raw_root: Path,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: torch.device | None = None,
    crop: bool = True,
    clip_weight: float | None = None,
    color_weight: float | None = None,
    color_bins: int | None = None,
) -> list[SimilarCard]:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve hybrid parameters: explicit args > sidecar file > built-in defaults
    saved = load_index_params(db_path)
    clip_weight = clip_weight if clip_weight is not None else saved["clip_weight"]
    color_weight = color_weight if color_weight is not None else saved["color_weight"]
    color_bins = color_bins if color_bins is not None else saved["color_bins"]

    dev = device or resolve_device(None)
    model, preprocess = load_clip(model_name, pretrained, dev)
    query_image = load_query_image(image_path, crop=crop)
    embedding = embed_image_hybrid(model, preprocess, query_image, dev, clip_weight, color_weight, color_bins)

    collection = open_chroma_collection(db_path, collection_name)
    if collection.count() == 0:
        raise RuntimeError(f"Collection {collection_name!r} is empty; run index first")

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(k, collection.count()),
        include=["metadatas", "distances"],
    )

    hits: list[SimilarCard] = []
    for card_id, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        set_id = meta["set_id"]
        raw_path = find_raw_image(card_id, set_id, raw_root)
        hits.append(
            SimilarCard(
                card_id=card_id,
                set_id=set_id,
                score=1.0 - dist,
                raw_path=raw_path,
                clean_path=meta.get("path"),
            )
        )
    return hits


def get_chroma_collection(db_path: Path, collection_name: str, reset: bool):
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    if reset:
        try:
            client.delete_collection(collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def cmd_index(args: argparse.Namespace) -> int:
    clean_root = args.clean_dir.expanduser().resolve()
    db_path = args.db_path.expanduser().resolve()

    if not clean_root.is_dir():
        print(f"Clean directory not found: {clean_root}", file=sys.stderr)
        return 1

    cards = discover_cards(clean_root)
    if not cards:
        print(f"No PNG files under {clean_root}", file=sys.stderr)
        return 1

    clip_weight = args.clip_weight if args.clip_weight is not None else DEFAULT_CLIP_WEIGHT
    color_weight = args.color_weight if args.color_weight is not None else DEFAULT_COLOR_WEIGHT
    color_bins = args.color_bins if args.color_bins is not None else DEFAULT_COLOR_BINS

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Loading CLIP {args.model}/{args.pretrained}...")
    model, preprocess = load_clip(args.model, args.pretrained, device)

    print(f"Embedding {len(cards)} cards (clip_weight={clip_weight}, color_weight={color_weight}, color_bins={color_bins})...")
    embeddings = embed_cards(
        model, preprocess, cards, device, args.batch_size,
        clip_weight=clip_weight,
        color_weight=color_weight,
        color_bins=color_bins,
    )

    collection = get_chroma_collection(db_path, args.collection, reset=args.force)
    ids = [c.card_id for c in cards]
    documents = [str(c.path.relative_to(_script_dir())) for c in cards]
    metadatas = [{"set_id": c.set_id, "path": documents[i]} for i, c in enumerate(cards)]

    # Chroma upsert handles create + update
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    save_index_params(db_path, clip_weight, color_weight, color_bins)

    print(
        f"Indexed {len(cards)} cards -> {db_path} "
        f"(collection={args.collection!r}, count={collection.count()})"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    raw_root = args.raw_dir.expanduser().resolve()
    db_path = args.db_path.expanduser().resolve()

    try:
        hits = query_similar(
            args.image,
            args.top_k,
            db_path=db_path,
            raw_root=raw_root,
            collection_name=args.collection,
            model_name=args.model,
            pretrained=args.pretrained,
            device=resolve_device(args.device),
            crop=not args.no_crop,
            clip_weight=args.clip_weight,
            color_weight=args.color_weight,
            color_bins=args.color_bins,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Query: {args.image.resolve()}\n")
    for rank, hit in enumerate(hits, start=1):
        raw = hit.raw_path if hit.raw_path else "(raw image not found)"
        print(
            f"{rank}. {hit.card_id}  set={hit.set_id}  "
            f"similarity={hit.score:.4f}\n   raw: {raw}"
        )
    return 0


def _add_clip_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Chroma persistence directory (default: <project>/data/chroma)",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help=f"OpenCLIP pretrained weights (default: {DEFAULT_PRETRAINED})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--clip-weight",
        type=float,
        default=None,
        help=f"Weight applied to CLIP embedding in hybrid vector (default: read from index_params.json, else {DEFAULT_CLIP_WEIGHT})",
    )
    parser.add_argument(
        "--color-weight",
        type=float,
        default=None,
        help=f"Weight applied to HSV color histogram in hybrid vector (default: read from index_params.json, else {DEFAULT_COLOR_WEIGHT})",
    )
    parser.add_argument(
        "--color-bins",
        type=int,
        default=None,
        help=f"Number of histogram bins per HSV channel (default: read from index_params.json, else {DEFAULT_COLOR_BINS})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIPeon CLIP indexing and search.")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Embed clean card art into ChromaDB")
    index.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Root of cleaned images (default: <project>/data/clean)",
    )
    _add_clip_args(index)
    index.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})",
    )
    index.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the collection before indexing",
    )

    query = sub.add_parser("query", help="Find k most similar indexed cards for an image")
    query.add_argument("image", type=Path, help="Query image (photo, crop, or card scan)")
    query.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of matches to return (default: 5)",
    )
    query.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw image root (default: <project>/data/raw)",
    )
    query.add_argument(
        "--no-crop",
        action="store_true",
        help="Skip layout detection/crop (use when input is already cleaned art)",
    )
    _add_clip_args(query)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "db_path", None) is None:
        args.db_path = _script_dir() / "data" / "chroma"
    if args.command == "index" and args.clean_dir is None:
        args.clean_dir = _script_dir() / "data" / "clean"
    if args.command == "query" and args.raw_dir is None:
        args.raw_dir = _script_dir() / "data" / "raw"

    if args.command == "index":
        return cmd_index(args)
    if args.command == "query":
        return cmd_query(args)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
