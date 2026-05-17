#!/usr/bin/env python3
"""
Crop card art from raw images into data/clean/<set_id>/.

Layout is chosen per rarity hint and/or image detection:
  - Standard (Common, Shiny Rare, …):      (30, 110, 704, 510)
  - Full-art borderless:                   (30,  30, 704, 870)
  - Full-art with large title bar/header:  (30, 110, 704, 870)

Usage:
  python clean_data.py            # all raw sets missing from data/clean
  python clean_data.py me2        # one set
  python clean_data.py me1 --force
  python clean_data.py --force    # re-clean every set in raw
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

# Always use standard art crop (same as Common / Uncommon / Rare).
STANDARD_RARITIES = frozenset({
    "Common",
    "Uncommon",
    "Rare",
    "Double Rare",
    "Shiny",
    "Shiny Rare",
})

# Full-art family: header vs borderless detection only (no silver-frame check).
FULL_ART_RARITIES = frozenset({
    "Ultra Rare",
    "Special Illustration Rare",
    "Mega Hyper Rare",
    "Hyper Rare",
    "Shiny Ultra Rare",
})


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _get_pixel(im: Image.Image, x: int, y: int) -> tuple[int, int, int]:
    return im.getpixel((min(x, im.width - 1), min(y, im.height - 1)))


def _has_silver_art_border(im: Image.Image) -> bool:
    w, h = im.size
    samples = ((15, min(200, h - 1)), (15, min(350, h - 1)), (min(718, w - 1), min(200, h - 1)))
    lums = [_luminance(_get_pixel(im, x, y)) for x, y in samples]
    return sum(lums) / len(lums) > SILVER_BORDER_LUM_THRESHOLD


def _header_row_luminance(im: Image.Image) -> float:
    x0, x1, step = 30, min(704, im.width), 50
    y = min(HEADER_PROBE_Y, im.height - 1)
    row = [_luminance(_get_pixel(im, x, y)) for x in range(x0, x1, step)]
    return sum(row) / len(row)


def _clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    left, top, right, bottom = box
    return (min(left, w), min(top, h), min(right, w), min(bottom, h))


def detect_art_box_full_art(im: Image.Image) -> tuple[int, int, int, int]:
    if _header_row_luminance(im) > HEADER_LUM_THRESHOLD:
        return FULL_HEADER_ART_BOX
    return FULL_BORDERLESS_ART_BOX


def detect_art_box(im: Image.Image, rarity: str | None = None) -> tuple[int, int, int, int]:
    if rarity in STANDARD_RARITIES:
        return STANDARD_ART_BOX
    if rarity in FULL_ART_RARITIES:
        return detect_art_box_full_art(im)
    if _has_silver_art_border(im):
        return STANDARD_ART_BOX
    return detect_art_box_full_art(im)


def crop_card_art(
    im: Image.Image, rarity: str | None = None
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Detect layout and return cropped art (same logic as the clean pipeline)."""
    rgb = im.convert("RGB")
    box = _clamp_box(detect_art_box(rgb, rarity=rarity), rgb.size)
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


def discover_set_ids(raw_root: Path, clean_root: Path, *, force: bool) -> list[str]:
    raw_sets = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    if force:
        return raw_sets
    return [sid for sid in raw_sets if not (clean_root / sid).is_dir()]


def process_set(
    in_root: Path,
    out_root: Path,
    *,
    suffix_filter: str | None,
    force: bool,
) -> tuple[int, int, int]:
    ok = 0
    skip = 0
    fail = 0

    for rarity_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        for src in sorted(rarity_dir.glob("*.png")):
            if suffix_filter and not src.stem.endswith(suffix_filter):
                continue

            dest = out_root / clean_name(src)
            if dest.is_file() and not force:
                skip += 1
                continue

            try:
                rarity = rarity_dir.name
                with Image.open(src) as im:
                    cropped, art_box = crop_card_art(im, rarity=rarity)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cropped.save(dest)
                print(f"{src.relative_to(in_root)} [{rarity}] {art_box} -> {dest.name}")
                ok += 1
            except OSError as e:
                print(f"Failed {src}: {e}", file=sys.stderr)
                fail += 1

    return ok, skip, fail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop card art from raw images (layout detected per card)."
    )
    parser.add_argument(
        "set_id",
        nargs="?",
        default=None,
        help="Set id under data/raw (default: all sets not yet in data/clean)",
    )
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

    raw_root = (_script_dir() / "data" / "raw").resolve()
    clean_root = (_script_dir() / "data" / "clean").resolve()

    if args.input is not None and args.set_id is None:
        print("set_id is required when using --input", file=sys.stderr)
        return 1

    suffix_filter: str | None
    if args.size == "large":
        suffix_filter = "_large"
    elif args.size == "small":
        suffix_filter = "_small"
    else:
        suffix_filter = None

    if args.set_id:
        set_ids = [args.set_id.strip().lower()]
    else:
        if not raw_root.is_dir():
            print(f"Raw directory not found: {raw_root}", file=sys.stderr)
            return 1
        set_ids = discover_set_ids(raw_root, clean_root, force=args.force)
        if not set_ids:
            print("No sets to process (all raw sets already have clean output)")
            return 0
        print(f"Processing {len(set_ids)} set(s): {', '.join(set_ids)}")

    total_ok = 0
    total_skip = 0
    total_fail = 0

    for set_id in set_ids:
        in_root = args.input or (raw_root / set_id)
        out_root = args.output or (clean_root / set_id)
        in_root = in_root.expanduser().resolve()
        out_root = out_root.expanduser().resolve()

        if not in_root.is_dir():
            print(f"Input directory not found: {in_root}", file=sys.stderr)
            total_fail += 1
            continue

        print(f"\n=== {set_id} ===")
        ok, skip, fail = process_set(
            in_root, out_root, suffix_filter=suffix_filter, force=args.force
        )
        print(f"Done {set_id}. cropped={ok} skipped={skip} failed={fail} -> {out_root}")
        total_ok += ok
        total_skip += skip
        total_fail += fail

    print(f"\nTotal. cropped={total_ok} skipped={total_skip} failed={total_fail}")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
