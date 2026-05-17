#!/usr/bin/env python3
"""
CLIPeon three-signal indexing and search.

Signals:
  - CLIP (ViT-B/32)        – semantic subject understanding
  - DINOv2 (ViT-B/14)     – visual style, texture, composition
  - HSV color histogram    – palette similarity

Each signal is stored in its own ChromaDB collection so weights can be tuned
independently at query time without re-indexing.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "openai"
DEFAULT_DINO_MODEL = "dinov2_vitb14"

CLIP_COLLECTION = "cards_clip"
DINO_COLLECTION = "cards_dino"
COLOR_COLLECTION = "cards_color"

# Legacy single-collection name kept so old CLI users get a clear error.
DEFAULT_COLLECTION = "cards"

EMBED_BATCH_SIZE = 32
DEFAULT_COLOR_BINS = 32
DEFAULT_CLIP_WEIGHT = 0.40
DEFAULT_DINO_WEIGHT = 0.40
DEFAULT_COLOR_WEIGHT = 0.20
DEFAULT_H_WEIGHT = 2.0
DEFAULT_S_WEIGHT = 1.0
DEFAULT_V_WEIGHT = 0.5

# DINOv2 image normalisation (ImageNet stats used by the official model).
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)
_DINO_RESIZE = 256
_DINO_CROP = 224


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

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
    clip_score: float
    dino_score: float
    color_score: float
    raw_path: Path | None
    clean_path: str | None


# ---------------------------------------------------------------------------
# Color histogram (shared utility)
# ---------------------------------------------------------------------------

def color_histogram(
    pil_img: Image.Image,
    bins: int = DEFAULT_COLOR_BINS,
    h_weight: float = DEFAULT_H_WEIGHT,
    s_weight: float = DEFAULT_S_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
) -> np.ndarray:
    """Compute a weighted HSV histogram from a PIL RGB image, L1-normalised.

    Output shape: (3*bins,).
    """
    hsv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0].astype(np.float32) * h_weight
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0].astype(np.float32) * s_weight
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0].astype(np.float32) * v_weight
    hist = np.concatenate([h_hist, s_hist, v_hist])
    total = hist.sum()
    return hist / (total + 1e-8)


def embed_color_vector(
    pil_img: Image.Image,
    bins: int = DEFAULT_COLOR_BINS,
    h_weight: float = DEFAULT_H_WEIGHT,
    s_weight: float = DEFAULT_S_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
) -> list[float]:
    """L2-normalised color histogram suitable for Chroma cosine metric."""
    hist = color_histogram(pil_img, bins=bins, h_weight=h_weight, s_weight=s_weight, v_weight=v_weight)
    norm = np.linalg.norm(hist) + 1e-8
    return (hist / norm).tolist()


# ---------------------------------------------------------------------------
# Index-params sidecar
# ---------------------------------------------------------------------------

def _script_dir() -> Path:
    return Path(__file__).resolve().parent


INDEX_PARAMS_FILE = "index_params.json"


def save_index_params(
    db_path: Path,
    clip_weight: float,
    dino_weight: float,
    color_weight: float,
    color_bins: int,
    h_weight: float,
    s_weight: float,
    v_weight: float,
    dino_model: str,
) -> None:
    params = {
        "clip_weight": clip_weight,
        "dino_weight": dino_weight,
        "color_weight": color_weight,
        "color_bins": color_bins,
        "h_weight": h_weight,
        "s_weight": s_weight,
        "v_weight": v_weight,
        "dino_model": dino_model,
    }
    (db_path / INDEX_PARAMS_FILE).write_text(json.dumps(params, indent=2))


def load_index_params(db_path: Path) -> dict:
    """Load saved index parameters, back-filling defaults for missing keys."""
    params_path = db_path / INDEX_PARAMS_FILE
    if params_path.is_file():
        saved = json.loads(params_path.read_text())
        saved.setdefault("dino_weight", DEFAULT_DINO_WEIGHT)
        saved.setdefault("color_weight", DEFAULT_COLOR_WEIGHT)
        saved.setdefault("h_weight", DEFAULT_H_WEIGHT)
        saved.setdefault("s_weight", DEFAULT_S_WEIGHT)
        saved.setdefault("v_weight", DEFAULT_V_WEIGHT)
        saved.setdefault("dino_model", DEFAULT_DINO_MODEL)
        return saved
    return {
        "clip_weight": DEFAULT_CLIP_WEIGHT,
        "dino_weight": DEFAULT_DINO_WEIGHT,
        "color_weight": DEFAULT_COLOR_WEIGHT,
        "color_bins": DEFAULT_COLOR_BINS,
        "h_weight": DEFAULT_H_WEIGHT,
        "s_weight": DEFAULT_S_WEIGHT,
        "v_weight": DEFAULT_V_WEIGHT,
        "dino_model": DEFAULT_DINO_MODEL,
    }


# ---------------------------------------------------------------------------
# Card discovery
# ---------------------------------------------------------------------------

def discover_cards(clean_root: Path) -> list[CardImage]:
    cards: list[CardImage] = []
    for set_dir in sorted(p for p in clean_root.iterdir() if p.is_dir()):
        for path in sorted(set_dir.glob("*.png")):
            cards.append(
                CardImage(card_id=path.stem, set_id=set_dir.name, path=path)
            )
    return cards


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


def load_dino(model_name: str = DEFAULT_DINO_MODEL, device: torch.device | None = None):
    """Load a DINOv2 model and its preprocessing transform via torch.hub.

    The model weights are cached in ~/.cache/torch/hub after the first download.
    """
    from torchvision import transforms

    dev = device or resolve_device(None)
    model = torch.hub.load(
        "facebookresearch/dinov2",
        model_name,
        pretrained=True,
        verbose=False,
    )
    model.eval()
    model.to(dev)

    transform = transforms.Compose([
        transforms.Resize(_DINO_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_DINO_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=_DINO_MEAN, std=_DINO_STD),
    ])
    return model, transform


# ---------------------------------------------------------------------------
# Dataset / DataLoader helper
# ---------------------------------------------------------------------------

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


def _default_collate(batch):
    return (
        torch.stack([b[0] for b in batch]),
        [b[1] for b in batch],   # PIL images
        [b[2] for b in batch],   # CardImage metadata
    )


# ---------------------------------------------------------------------------
# Batch embedding: CLIP
# ---------------------------------------------------------------------------

def embed_cards_clip(
    model,
    preprocess,
    cards: list[CardImage],
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    """Embed cards with CLIP only (no color mixing). Output: L2-normalised 512-dim."""
    if not cards:
        return []
    dataset = _ImageDataset(cards, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=_default_collate)
    vectors: list[list[float]] = []
    with torch.inference_mode():
        for images, _pils, _cards in loader:
            images = images.to(device)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.extend(feats.cpu().tolist())
    return vectors


# ---------------------------------------------------------------------------
# Batch embedding: DINOv2
# ---------------------------------------------------------------------------

def embed_cards_dino(
    model,
    transform,
    cards: list[CardImage],
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    """Embed cards with DINOv2 only. Output: L2-normalised 768-dim CLS token."""
    if not cards:
        return []
    dataset = _ImageDataset(cards, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=_default_collate)
    vectors: list[list[float]] = []
    with torch.inference_mode():
        for images, _pils, _cards in loader:
            images = images.to(device)
            feats = model(images)
            # torch.hub dinov2_vitb14 returns (B, 768) CLS token directly
            if isinstance(feats, dict):
                feats = feats["x_norm_clstoken"]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.extend(feats.cpu().tolist())
    return vectors


# ---------------------------------------------------------------------------
# Batch embedding: color histograms
# ---------------------------------------------------------------------------

def embed_cards_color(
    cards: list[CardImage],
    bins: int = DEFAULT_COLOR_BINS,
    h_weight: float = DEFAULT_H_WEIGHT,
    s_weight: float = DEFAULT_S_WEIGHT,
    v_weight: float = DEFAULT_V_WEIGHT,
) -> list[list[float]]:
    """Compute L2-normalised HSV histogram vectors for all cards (CPU, fast)."""
    vectors: list[list[float]] = []
    for card in cards:
        pil_img = Image.open(card.path).convert("RGB")
        vectors.append(embed_color_vector(pil_img, bins, h_weight, s_weight, v_weight))
    return vectors


# ---------------------------------------------------------------------------
# Single-image embedding (query time)
# ---------------------------------------------------------------------------

def load_query_image(image_path: Path, *, crop: bool = True) -> Image.Image:
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")
    if crop and is_full_card_image(rgb):
        cropped, _box = crop_card_art(rgb)
        return cropped
    return rgb


def embed_image_clip(
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


def embed_image_dino(
    model,
    transform,
    image: Image.Image,
    device: torch.device,
) -> list[float]:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feats = model(tensor)
        if isinstance(feats, dict):
            feats = feats["x_norm_clstoken"]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().tolist()[0]


# ---------------------------------------------------------------------------
# ChromaDB helpers
# ---------------------------------------------------------------------------

def find_raw_image(card_id: str, set_id: str, raw_root: Path) -> Path | None:
    set_dir = raw_root / set_id
    if not set_dir.is_dir():
        return None
    matches = sorted(set_dir.glob(f"**/{card_id}_large.png"))
    return matches[0] if matches else None


def open_chroma_collection(db_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_collection(name=collection_name)


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


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def _fuse_scores(
    clip_results: dict,
    dino_results: dict,
    color_results: dict,
    w_clip: float,
    w_dino: float,
    w_color: float,
) -> list[tuple[str, float, float, float, float]]:
    """Merge ranked lists from three collections into one fused ranking.

    Returns a list of (card_id, fused_score, clip_score, dino_score, color_score)
    sorted descending by fused_score.  Absent scores default to 0.
    """
    clip_scores: dict[str, float] = {
        id_: 1.0 - dist
        for id_, dist in zip(clip_results["ids"][0], clip_results["distances"][0])
    }
    dino_scores: dict[str, float] = {
        id_: 1.0 - dist
        for id_, dist in zip(dino_results["ids"][0], dino_results["distances"][0])
    }
    color_scores: dict[str, float] = {
        id_: 1.0 - dist
        for id_, dist in zip(color_results["ids"][0], color_results["distances"][0])
    }

    all_ids = set(clip_scores) | set(dino_scores) | set(color_scores)
    fused: list[tuple[str, float, float, float, float]] = []
    for card_id in all_ids:
        cs = clip_scores.get(card_id, 0.0)
        ds = dino_scores.get(card_id, 0.0)
        co = color_scores.get(card_id, 0.0)
        fused.append((card_id, w_clip * cs + w_dino * ds + w_color * co, cs, ds, co))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# ---------------------------------------------------------------------------
# Public query API
# ---------------------------------------------------------------------------

def query_similar(
    image_path: Path,
    k: int,
    *,
    db_path: Path,
    raw_root: Path,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    dino_model_name: str | None = None,
    device: torch.device | None = None,
    crop: bool = True,
    clip_weight: float | None = None,
    dino_weight: float | None = None,
    color_weight: float | None = None,
    color_bins: int | None = None,
    h_weight: float | None = None,
    s_weight: float | None = None,
    v_weight: float | None = None,
) -> list[SimilarCard]:
    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resolve parameters: explicit args > sidecar file > built-in defaults
    saved = load_index_params(db_path)
    clip_weight = clip_weight if clip_weight is not None else saved["clip_weight"]
    dino_weight = dino_weight if dino_weight is not None else saved["dino_weight"]
    color_weight = color_weight if color_weight is not None else saved["color_weight"]
    color_bins = color_bins if color_bins is not None else saved["color_bins"]
    h_weight = h_weight if h_weight is not None else saved["h_weight"]
    s_weight = s_weight if s_weight is not None else saved["s_weight"]
    v_weight = v_weight if v_weight is not None else saved["v_weight"]
    dino_model_name = dino_model_name or saved["dino_model"]

    dev = device or resolve_device(None)

    clip_model, preprocess = load_clip(model_name, pretrained, dev)
    dino_model, dino_transform = load_dino(dino_model_name, dev)

    query_image = load_query_image(image_path, crop=crop)
    q_clip = embed_image_clip(clip_model, preprocess, query_image, dev)
    q_dino = embed_image_dino(dino_model, dino_transform, query_image, dev)
    q_color = embed_color_vector(query_image, color_bins, h_weight, s_weight, v_weight)

    clip_col = open_chroma_collection(db_path, CLIP_COLLECTION)
    dino_col = open_chroma_collection(db_path, DINO_COLLECTION)
    color_col = open_chroma_collection(db_path, COLOR_COLLECTION)

    total = clip_col.count()
    if total == 0:
        raise RuntimeError("Collections are empty; run: python clip_actions.py index")

    # Over-fetch so late fusion can surface cards that scored high in ≥1 signal
    # but not necessarily top-k in all three.
    candidate_n = min(k * 10, total)

    clip_results = clip_col.query(query_embeddings=[q_clip], n_results=candidate_n,
                                  include=["metadatas", "distances"])
    dino_results = dino_col.query(query_embeddings=[q_dino], n_results=candidate_n,
                                  include=["distances"])
    color_results = color_col.query(query_embeddings=[q_color], n_results=candidate_n,
                                    include=["distances"])

    fused = _fuse_scores(clip_results, dino_results, color_results,
                         clip_weight, dino_weight, color_weight)

    # Metadata lives in the clip collection (primary)
    meta_lookup: dict[str, dict] = {
        id_: meta
        for id_, meta in zip(clip_results["ids"][0], clip_results["metadatas"][0])
    }
    # Back-fill any fused IDs not in the clip candidate pool
    missing_ids = [t[0] for t in fused[:k] if t[0] not in meta_lookup]
    if missing_ids:
        extra = clip_col.get(ids=missing_ids, include=["metadatas"])
        for id_, meta in zip(extra["ids"], extra["metadatas"]):
            meta_lookup[id_] = meta

    hits: list[SimilarCard] = []
    for card_id, fused_score, cs, ds, co in fused[:k]:
        meta = meta_lookup.get(card_id, {})
        set_id = meta.get("set_id", "")
        raw_path = find_raw_image(card_id, set_id, raw_root)
        hits.append(SimilarCard(
            card_id=card_id,
            set_id=set_id,
            score=fused_score,
            clip_score=cs,
            dino_score=ds,
            color_score=co,
            raw_path=raw_path,
            clean_path=meta.get("path"),
        ))
    return hits


# ---------------------------------------------------------------------------
# CLI: index
# ---------------------------------------------------------------------------

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
    dino_weight = args.dino_weight if args.dino_weight is not None else DEFAULT_DINO_WEIGHT
    color_weight = args.color_weight if args.color_weight is not None else DEFAULT_COLOR_WEIGHT
    color_bins = args.color_bins if args.color_bins is not None else DEFAULT_COLOR_BINS
    h_weight = args.h_weight if args.h_weight is not None else DEFAULT_H_WEIGHT
    s_weight = args.s_weight if args.s_weight is not None else DEFAULT_S_WEIGHT
    v_weight = args.v_weight if args.v_weight is not None else DEFAULT_V_WEIGHT
    dino_model_name = args.dino_model if args.dino_model is not None else DEFAULT_DINO_MODEL

    device = resolve_device(args.device)
    print(f"Device: {device}")

    print(f"Loading CLIP {args.model}/{args.pretrained}...")
    clip_model, preprocess = load_clip(args.model, args.pretrained, device)

    print(f"Loading DINOv2 {dino_model_name} (downloads on first run)...")
    dino_model, dino_transform = load_dino(dino_model_name, device)

    print(f"Embedding {len(cards)} cards with CLIP...")
    clip_vecs = embed_cards_clip(clip_model, preprocess, cards, device, args.batch_size)

    print(f"Embedding {len(cards)} cards with DINOv2...")
    dino_vecs = embed_cards_dino(dino_model, dino_transform, cards, device, args.batch_size)

    print(f"Computing color histograms (bins={color_bins})...")
    color_vecs = embed_cards_color(cards, color_bins, h_weight, s_weight, v_weight)

    ids = [c.card_id for c in cards]
    documents = [str(c.path.relative_to(_script_dir())) for c in cards]
    metadatas = [{"set_id": c.set_id, "path": documents[i]} for i, c in enumerate(cards)]

    print("Upserting into ChromaDB collections...")
    clip_col = get_chroma_collection(db_path, CLIP_COLLECTION, reset=args.force)
    dino_col = get_chroma_collection(db_path, DINO_COLLECTION, reset=args.force)
    color_col = get_chroma_collection(db_path, COLOR_COLLECTION, reset=args.force)

    clip_col.upsert(ids=ids, embeddings=clip_vecs, metadatas=metadatas, documents=documents)
    dino_col.upsert(ids=ids, embeddings=dino_vecs, metadatas=metadatas, documents=documents)
    color_col.upsert(ids=ids, embeddings=color_vecs, metadatas=metadatas, documents=documents)

    save_index_params(
        db_path, clip_weight, dino_weight, color_weight,
        color_bins, h_weight, s_weight, v_weight, dino_model_name,
    )

    print(
        f"Indexed {len(cards)} cards -> {db_path}\n"
        f"  clip_weight={clip_weight}, dino_weight={dino_weight}, color_weight={color_weight}\n"
        f"  color_bins={color_bins}, h={h_weight}, s={s_weight}, v={v_weight}\n"
        f"  dino_model={dino_model_name}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI: query
# ---------------------------------------------------------------------------

def cmd_query(args: argparse.Namespace) -> int:
    raw_root = args.raw_dir.expanduser().resolve()
    db_path = args.db_path.expanduser().resolve()

    try:
        hits = query_similar(
            args.image,
            args.top_k,
            db_path=db_path,
            raw_root=raw_root,
            model_name=args.model,
            pretrained=args.pretrained,
            dino_model_name=args.dino_model,
            device=resolve_device(args.device),
            crop=not args.no_crop,
            clip_weight=args.clip_weight,
            dino_weight=args.dino_weight,
            color_weight=args.color_weight,
            color_bins=args.color_bins,
            h_weight=args.h_weight,
            s_weight=args.s_weight,
            v_weight=args.v_weight,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(e, file=sys.stderr)
        return 1

    print(f"Query: {args.image.resolve()}\n")
    for rank, hit in enumerate(hits, start=1):
        raw = hit.raw_path if hit.raw_path else "(raw image not found)"
        print(
            f"{rank}. {hit.card_id}  set={hit.set_id}  "
            f"score={hit.score:.4f}  "
            f"(clip={hit.clip_score:.3f} dino={hit.dino_score:.3f} color={hit.color_score:.3f})\n"
            f"   raw: {raw}"
        )
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Chroma persistence directory (default: <project>/data/chroma)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenCLIP model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED,
                        help=f"OpenCLIP pretrained weights (default: {DEFAULT_PRETRAINED})")
    parser.add_argument("--dino-model", default=None,
                        help=f"DINOv2 torch.hub model name (default: {DEFAULT_DINO_MODEL})")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--clip-weight", type=float, default=None,
                        help=f"CLIP signal weight (default: from index_params.json, else {DEFAULT_CLIP_WEIGHT})")
    parser.add_argument("--dino-weight", type=float, default=None,
                        help=f"DINOv2 signal weight (default: from index_params.json, else {DEFAULT_DINO_WEIGHT})")
    parser.add_argument("--color-weight", type=float, default=None,
                        help=f"Color histogram signal weight (default: from index_params.json, else {DEFAULT_COLOR_WEIGHT})")
    parser.add_argument("--color-bins", type=int, default=None,
                        help=f"HSV histogram bins per channel (default: {DEFAULT_COLOR_BINS})")
    parser.add_argument("--h-weight", type=float, default=None,
                        help=f"Hue channel scale (default: {DEFAULT_H_WEIGHT})")
    parser.add_argument("--s-weight", type=float, default=None,
                        help=f"Saturation channel scale (default: {DEFAULT_S_WEIGHT})")
    parser.add_argument("--v-weight", type=float, default=None,
                        help=f"Value channel scale (default: {DEFAULT_V_WEIGHT})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLIPeon three-signal card art search.")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Embed card art into three ChromaDB collections")
    index.add_argument("--clean-dir", type=Path, default=None,
                       help="Root of cleaned images (default: <project>/data/clean)")
    _add_shared_args(index)
    index.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE,
                       help=f"Embedding batch size (default: {EMBED_BATCH_SIZE})")
    index.add_argument("--force", action="store_true",
                       help="Drop and recreate all three collections before indexing")

    query = sub.add_parser("query", help="Find k most similar cards using three-signal fusion")
    query.add_argument("image", type=Path, help="Query image (photo, crop, or card scan)")
    query.add_argument("-k", "--top-k", type=int, default=5,
                       help="Number of matches to return (default: 5)")
    query.add_argument("--raw-dir", type=Path, default=None,
                       help="Raw image root (default: <project>/data/raw)")
    query.add_argument("--no-crop", action="store_true",
                       help="Skip layout detection/crop (use when input is already cleaned art)")
    _add_shared_args(query)

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
