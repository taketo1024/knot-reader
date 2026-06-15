"""Stage 1: rasterize a PDF/image and split it into individual knot diagrams.

Each knot is one large blob of ink; labels and captions are tiny components.
We detect the big blobs (after a morphological close that bridges the
over/under gaps) and crop each one from the *original* binary so those gaps
survive for later over/under detection.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Diagram:
    """One cropped knot diagram."""
    name: str          # e.g. "diagram-0" (left-to-right order)
    bbox: tuple        # (x, y, w, h) in the source image
    binary: np.ndarray  # cropped binary, 255 = ink


def rasterize_pdf(pdf: Path, page: int, dpi: int, out_dir: Path) -> Path:
    """Render one PDF page to PNG via pdftoppm; return the PNG path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"page{page}"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page),
         str(pdf), str(prefix)],
        check=True,
    )
    # pdftoppm appends "-<page>" (zero-padded for multi-page); find the result.
    matches = sorted(out_dir.glob(f"page{page}-*.png"))
    if not matches:
        raise FileNotFoundError(f"pdftoppm produced no PNG for page {page}")
    return matches[0]


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold to a binary image with 255 = ink."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def find_diagrams(bw: np.ndarray, close_px: int = 25,
                  min_area_frac: float = 0.01, pad: int = 20) -> list[Diagram]:
    """Locate knot diagrams as the large connected components of the ink.

    A morphological close bridges line-breaks/dashes so each knot is a single
    component for detection; crops are taken from the un-closed `bw`.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)

    min_area = min_area_frac * bw.shape[0] * bw.shape[1]
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    keep.sort(key=lambda i: stats[i, cv2.CC_STAT_LEFT])  # left-to-right

    diagrams = []
    H, W = bw.shape
    for idx, i in enumerate(keep):
        x, y, w, h, _ = stats[i]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        diagrams.append(Diagram(
            name=f"diagram-{idx}",
            bbox=(x0, y0, x1 - x0, y1 - y0),
            binary=bw[y0:y1, x0:x1].copy(),
        ))
    return diagrams


def load_gray(path: Path, dpi: int = 300, page: int = 1) -> np.ndarray:
    """Read any supported input (pdf rasterized at `dpi`, else image) as gray."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        png = rasterize_pdf(path, page, dpi, Path(tempfile.mkdtemp()))
        path = png
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"could not read image: {path}")
    return gray


def diagrams_from_path(path: Path, dpi: int = 300):
    """Load any input and return (binary_full, [Diagram, ...])."""
    binary = binarize(load_gray(path, dpi))
    return binary, find_diagrams(binary)


def split_page(pdf: Path, page: int = 1, dpi: int = 300,
               out_dir: Path = Path("out")) -> list[Diagram]:
    """End-to-end stage 1: PDF page -> list of cropped knot diagrams."""
    png = rasterize_pdf(pdf, page, dpi, out_dir)
    gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    return find_diagrams(binarize(gray))
