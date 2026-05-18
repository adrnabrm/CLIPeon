#!/usr/bin/env python3
"""
Download card images listed in pokemon-tcg-data JSON into data/raw/<set_id>/<rarity>/.

Usage:
  python gather_data.py me1
  python gather_data.py me1 me2 svp --size large

Resolves the set JSON by trying, in order:
  - $POKEMON_TCG_CARDS_DIR/<set_id>.json if the env var is set
  - ~/Projects/pokemon-tcg-data/en/cards/<set_id>.json
  - ~/Projects/pokemon-tcg-data/cards/en/<set_id>.json
  - <repo>/../pokemon-tcg-data/en/cards/ and cards/en (same filenames)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_json_path(set_id: str) -> Path:
    env = os.environ.get("POKEMON_TCG_CARDS_DIR")
    if env:
        p = Path(env).expanduser() / f"{set_id}.json"
        if p.is_file():
            return p
        raise FileNotFoundError(
            f"POKEMON_TCG_CARDS_DIR is set but missing file: {p}"
        )

    base = _script_dir().parent
    home_data = Path.home() / "Projects" / "pokemon-tcg-data"
    candidates = [
        home_data / "en" / "cards" / f"{set_id}.json",
        home_data / "cards" / "en" / f"{set_id}.json",
        base / "pokemon-tcg-data" / "en" / "cards" / f"{set_id}.json",
        base / "pokemon-tcg-data" / "cards" / "en" / f"{set_id}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No JSON found for set {set_id!r}. Tried:\n  {tried}\n"
        "Set POKEMON_TCG_CARDS_DIR to the folder that contains <set>.json"
    )


def rarity_subdir(rarity: object) -> str:
    if not rarity or not isinstance(rarity, str):
        return "Unknown"
    s = rarity.strip()
    if not s:
        return "Unknown"
    for ch in "/\\:\0\n\r\t":
        s = s.replace(ch, "-")
    return s


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CLIPeon-gather_data/1.0 (educational; +https://github.com/)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())


def download_set(
    set_id: str,
    out_root: Path,
    sizes: list[tuple[str, str]],
) -> tuple[int, int]:
    """Download all cards for one set. Returns (ok_count, fail_count)."""
    set_id = set_id.strip().lower()
    json_path = resolve_json_path(set_id)

    with json_path.open(encoding="utf-8") as f:
        cards = json.load(f)

    if not isinstance(cards, list):
        print(f"Expected a JSON array in {json_path}", file=sys.stderr)
        return 0, 1

    ok = 0
    fail = 0
    for card in cards:
        cid = card.get("id")
        images = card.get("images") or {}
        if not cid:
            print("Skipping entry without id", file=sys.stderr)
            fail += 1
            continue
        for label, key in sizes:
            url = images.get(key)
            if not url:
                print(f"No {key} URL for {cid}", file=sys.stderr)
                fail += 1
                continue
            suffix = Path(urlparse(url).path).suffix or ".png"
            sub = rarity_subdir(card.get("rarity"))
            dest = out_root / sub / f"{cid}_{label}{suffix}"
            if dest.is_file():
                ok += 1
                continue
            try:
                rel = dest.relative_to(out_root)
                print(f"GET {url} -> {rel}")
                download(url, dest)
                ok += 1
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"Failed {cid} ({label}): {e}", file=sys.stderr)
                fail += 1

    print(f"Done {set_id}. ok={ok} failed={fail} -> {out_root}")
    return ok, fail


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Pokémon TCG card images from set JSON.")
    parser.add_argument(
        "set_ids",
        nargs="+",
        metavar="set_id",
        help="One or more set ids matching JSON basenames, e.g. me1 me2 svp",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory for a single set (default: <project>/data/raw/<set_id>)",
    )
    parser.add_argument(
        "--size",
        choices=("small", "large", "both"),
        default="both",
        help="Which image URLs to download (default: both)",
    )
    args = parser.parse_args()

    set_ids = [s.strip().lower() for s in args.set_ids]
    if not set_ids:
        print("At least one set_id is required", file=sys.stderr)
        return 1

    if args.output is not None and len(set_ids) > 1:
        print(
            "Use -o with a single set_id, or omit -o to write each set under data/raw/<set_id>/",
            file=sys.stderr,
        )
        return 1

    sizes: list[tuple[str, str]] = []
    if args.size in ("small", "both"):
        sizes.append(("small", "small"))
    if args.size in ("large", "both"):
        sizes.append(("large", "large"))

    total_ok = 0
    total_fail = 0
    sets_failed = 0

    for set_id in set_ids:
        if args.output is None:
            out_root = _script_dir() / "data" / "raw" / set_id
        else:
            out_root = args.output.expanduser().resolve()

        print(f"\n=== {set_id} ===")
        try:
            ok, fail = download_set(set_id, out_root, sizes)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sets_failed += 1
            continue

        total_ok += ok
        total_fail += fail

    print(f"\nTotal. ok={total_ok} failed={total_fail}")
    if sets_failed:
        print(f"Sets skipped (missing JSON): {sets_failed}", file=sys.stderr)
    return 0 if total_fail == 0 and sets_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
