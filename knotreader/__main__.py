"""CLI: PDF/page -> PD code(s) + verification overlay(s).

    python -m knotreader exotic_disk.pdf --page 1 --dpi 300 --out out
"""

import argparse
from pathlib import Path

import cv2

from .split import split_page
from .trace import diagram_to_pd, validate_pd
from .overlay import draw_overlay


def sort_pd(PD):
    """Order crossings by first entry `a` (incoming under-strand) ascending."""
    return sorted(PD, key=lambda q: q[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diagrams = split_page(args.pdf, args.page, args.dpi, args.out)
    print(f"found {len(diagrams)} diagram(s)")

    for d in diagrams:
        PD, info = diagram_to_pd(d.binary)
        PD = sort_pd(PD)
        gray = 255 - d.binary
        vis = draw_overlay(gray, info)
        overlay_path = args.out / f"{d.name}_overlay.png"
        cv2.imwrite(str(overlay_path), vis)

        print(f"\n=== {d.name}: {len(PD)} crossings  valid={validate_pd(PD)} ===")
        print("[" + ",".join(str(q) for q in PD) + "]")
        print(f"overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
