#!/usr/bin/env python3
"""
CLIP indexing and search for cleaned Pokémon TCG card art.

Phase 1 (index): embed all images under data/clean and persist to ChromaDB.

Usage (conda env clipeon):
  python clip_actions.py index
  python clip_actions.py index --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import chromadb
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
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


@dataclass(frozen=True)
class CardImage:
    card_id: str
    set_id: str
    path: Path


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


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
        return self.preprocess(image), card


def embed_cards(
    model,
    preprocess,
    cards: list[CardImage],
    device: torch.device,
    batch_size: int,
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
            [b[1] for b in batch],
        ),
    )

    vectors: list[list[float]] = []
    with torch.inference_mode():
        for images, _batch_cards in loader:
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.extend(feats.cpu().tolist())
    return vectors


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

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Loading CLIP {args.model}/{args.pretrained}...")
    model, preprocess = load_clip(args.model, args.pretrained, device)

    print(f"Embedding {len(cards)} cards...")
    embeddings = embed_cards(
        model, preprocess, cards, device, args.batch_size
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

    print(
        f"Indexed {len(cards)} cards -> {db_path} "
        f"(collection={args.collection!r}, count={collection.count()})"
    )
    return 0


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
    index.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Chroma persistence directory (default: <project>/data/chroma)",
    )
    index.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION})",
    )
    index.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})",
    )
    index.add_argument(
        "--pretrained",
        default=DEFAULT_PRETRAINED,
        help=f"OpenCLIP pretrained weights (default: {DEFAULT_PRETRAINED})",
    )
    index.add_argument(
        "--batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})",
    )
    index.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda if available else cpu)",
    )
    index.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the collection before indexing",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.clean_dir is None:
        args.clean_dir = _script_dir() / "data" / "clean"
    if args.db_path is None:
        args.db_path = _script_dir() / "data" / "chroma"

    if args.command == "index":
        return cmd_index(args)

    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
