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


def pdf_page_count(pdf: Path) -> int:
    """Number of pages in a PDF (via pdfinfo)."""
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.lower().startswith("pages:"):
            return int(line.split(":", 1)[1])
    raise ValueError("could not determine PDF page count")


def load_gray(path: Path, dpi: int = 300, page: int = 1) -> np.ndarray:
    """Read any input (single-page pdf rasterized at `dpi`, else image) as gray.

    Handles color and transparent (RGBA) PNGs: alpha is composited onto white
    so a transparent background reads as paper, not ink.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        n = pdf_page_count(path)
        if n != 1:
            raise ValueError(
                f"PDF has {n} pages; please supply a single-page PDF or an image.")
        png = rasterize_pdf(path, page, dpi, Path(tempfile.mkdtemp()))
        path = png
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:  # composite RGBA onto white
        a = img[..., 3:4].astype(float) / 255.0
        img = (img[..., :3].astype(float) * a + 255.0 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _upscale_gray(gray_crop: np.ndarray, target: int = 1622,
                  upscale_below: int = 1400) -> np.ndarray:
    """Upscale a small grayscale crop so its long side is ~`target` px.

    The tracing thresholds are tuned for ~300-dpi rasterization (long side
    ~1600). Upscaling the *grayscale* (then thresholding) preserves the thin
    under-strand gaps; upscaling the binary would bleed adjacent strands
    together. No-op once the long side is already >= upscale_below.
    """
    h, w = gray_crop.shape
    long_side = max(h, w)
    if long_side >= upscale_below:
        return gray_crop
    s = target / long_side
    return cv2.resize(gray_crop, (round(w * s), round(h * s)), interpolation=cv2.INTER_CUBIC)


def diagrams_from_path(path: Path, dpi: int = 300):
    """Load any input and return (binary_full, [Diagram, ...]).

    Small diagrams are re-binarized from an upscaled grayscale crop so the
    tracing thresholds (tuned for ~300 dpi) apply; large ones are untouched.
    """
    gray = load_gray(path, dpi)
    binary = binarize(gray)
    diagrams = find_diagrams(binary)
    for d in diagrams:
        x, y, w, h = d.bbox
        if max(w, h) < 1400:
            d.binary = binarize(_upscale_gray(gray[y:y + h, x:x + w]))
    return binary, diagrams


def split_page(pdf: Path, page: int = 1, dpi: int = 300,
               out_dir: Path = Path("out")) -> list[Diagram]:
    """End-to-end stage 1: PDF page -> list of cropped knot diagrams."""
    png = rasterize_pdf(pdf, page, dpi, out_dir)
    gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    return find_diagrams(binarize(gray))
