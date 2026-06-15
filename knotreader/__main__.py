"""CLI: PDF/page -> PD code(s) + verification overlay(s).

    python -m knotreader exotic_disk.pdf --page 1 --dpi 300 --out out
"""

import argparse
from pathlib import Path

import cv2

from .split import diagrams_from_path
from .trace import diagram_to_pd, validate_pd, sort_pd, gauss_code
from .overlay import draw_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="image (png/jpg) or single-page PDF")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--excl", type=float, default=8.0)
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    _, diagrams = diagrams_from_path(args.input, args.dpi)
    print(f"found {len(diagrams)} diagram(s)")

    for d in diagrams:
        PD, info = diagram_to_pd(d.binary, excl=args.excl)
        PD = sort_pd(PD)
        vis = draw_overlay(255 - d.binary, info)
        overlay_path = args.out / f"{d.name}_overlay.png"
        cv2.imwrite(str(overlay_path), vis)

        print(f"\n=== {d.name}: {len(PD)} crossings  valid={validate_pd(PD)} ===")
        print("Gauss: " + " ".join(gauss_code(info)))
        print("PD:    [" + ",".join(str(q) for q in PD) + "]")
        print(f"overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
