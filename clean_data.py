#!/usr/bin/env python3
"""
Crop card art from raw images into data/clean/<set_id>/.

Layout is detected per image:
  - Standard (silver art frame):           (30, 110, 704, 510)
  - Full-art borderless:                   (30,  30, 704, 870)
  - Full-art with large title bar/header:  (30, 110, 704, 870)

Usage:
  python clean_data.py me2
  python clean_data.py me1 --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: pip install pillow", file=sys.stderr)
    raise SystemExit(1)

STANDARD_ART_BOX = (30, 110, 704, 510)
FULL_BORDERLESS_ART_BOX = (30, 30, 704, 870)
FULL_HEADER_ART_BOX = (30, 110, 704, 870)

HEADER_PROBE_Y = 80
HEADER_LUM_THRESHOLD = 200
SILVER_BORDER_LUM_THRESHOLD = 180


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _has_silver_art_border(im: Image.Image) -> bool:
    samples = ((15, 200), (15, 350), (718, 200))
    lums = [_luminance(im.getpixel(p)) for p in samples]
    return sum(lums) / len(lums) > SILVER_BORDER_LUM_THRESHOLD


def _header_row_luminance(im: Image.Image) -> float:
    x0, x1, step = 30, 704, 50
    row = [_luminance(im.getpixel((x, HEADER_PROBE_Y))) for x in range(x0, x1, step)]
    return sum(row) / len(row)


def detect_art_box(im: Image.Image) -> tuple[int, int, int, int]:
    if _has_silver_art_border(im):
        return STANDARD_ART_BOX
    if _header_row_luminance(im) > HEADER_LUM_THRESHOLD:
        return FULL_HEADER_ART_BOX
    return FULL_BORDERLESS_ART_BOX


def crop_card_art(im: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Detect layout and return cropped art (same logic as the clean pipeline)."""
    rgb = im.convert("RGB")
    box = detect_art_box(rgb)
    return rgb.crop(box), box


def is_full_card_image(im: Image.Image) -> bool:
    """True when the image is large enough to be an uncropped TCG card scan."""
    return im.width >= 700 and im.height >= 900


def clean_name(path: Path) -> str:
    """me2-1_large.png -> me2-1.png"""
    stem = path.stem
    if stem.endswith("_large"):
        stem = stem[: -len("_large")]
    elif stem.endswith("_small"):
        stem = stem[: -len("_small")]
    return f"{stem}{path.suffix}"


def crop_card(im: Image.Image, dest: Path, box: tuple[int, int, int, int]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.crop(box).save(dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop card art from raw images (layout detected per card)."
    )
    parser.add_argument("set_id", help="Set id under data/raw, e.g. me2")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help="Raw input directory (default: <project>/data/raw/<set_id>)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Clean output directory (default: <project>/data/clean/<set_id>)",
    )
    parser.add_argument(
        "--size",
        choices=("large", "small", "all"),
        default="large",
        help="Which raw filenames to process (default: large)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing clean images",
    )
    args = parser.parse_args()

    set_id = args.set_id.strip().lower()
    in_root = args.input or (_script_dir() / "data" / "raw" / set_id)
    out_root = args.output or (_script_dir() / "data" / "clean" / set_id)
    in_root = in_root.expanduser().resolve()
    out_root = out_root.expanduser().resolve()

    if not in_root.is_dir():
        print(f"Input directory not found: {in_root}", file=sys.stderr)
        return 1

    suffix_filter: str | None
    if args.size == "large":
        suffix_filter = "_large"
    elif args.size == "small":
        suffix_filter = "_small"
    else:
        suffix_filter = None

    ok = 0
    skip = 0
    fail = 0

    for rarity_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        for src in sorted(rarity_dir.glob("*.png")):
            if suffix_filter and not src.stem.endswith(suffix_filter):
                continue

            dest = out_root / clean_name(src)
            if dest.is_file() and not args.force:
                skip += 1
                continue

            try:
                with Image.open(src) as im:
                    cropped, art_box = crop_card_art(im)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cropped.save(dest)
                print(f"{src.relative_to(in_root)} {art_box} -> {dest.name}")
                ok += 1
            except OSError as e:
                print(f"Failed {src}: {e}", file=sys.stderr)
                fail += 1

    print(f"Done. cropped={ok} skipped={skip} failed={fail} -> {out_root}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
