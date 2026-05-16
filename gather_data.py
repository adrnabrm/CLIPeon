#!/usr/bin/env python3
"""
Download card images listed in pokemon-tcg-data JSON into data/raw/<set_id>/<rarity>/.

Usage:
  python gather_data.py me1

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Pokémon TCG card images from set JSON.")
    parser.add_argument("set_id", help="Set id matching the JSON basename, e.g. me1")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <project>/data/raw/<set_id>, with rarity subfolders)",
    )
    parser.add_argument(
        "--size",
        choices=("small", "large", "both"),
        default="both",
        help="Which image URLs to download (default: both)",
    )
    args = parser.parse_args()

    set_id = args.set_id.strip().lower()
    json_path = resolve_json_path(set_id)

    out_root = args.output
    if out_root is None:
        out_root = _script_dir() / "data" / "raw" / set_id
    else:
        out_root = out_root.expanduser().resolve()

    with json_path.open(encoding="utf-8") as f:
        cards = json.load(f)

    if not isinstance(cards, list):
        print(f"Expected a JSON array in {json_path}", file=sys.stderr)
        return 1

    sizes: list[tuple[str, str]] = []
    if args.size in ("small", "both"):
        sizes.append(("small", "small"))
    if args.size in ("large", "both"):
        sizes.append(("large", "large"))

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

    print(f"Done. ok={ok} failed={fail} -> {out_root}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
