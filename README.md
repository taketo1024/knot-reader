# knot-reader

Reads a knot-diagram image (PNG / JPG / PDF) and extracts its **Gauss code** and
**PD code**, with a verification overlay (circled crossings, numbered edges).

The drawing convention it expects: the **under**-strand is broken (a gap, optionally
with short dashes) at each crossing; the **over**-strand is drawn solid-continuous.

## How it works

1. **split** — rasterize (PDF) and crop each knot as a large connected ink component.
2. **trace** — skeletonize → ordered arcs → pair the broken ends into *bridges* by
   velocity continuity → crossings = bridge × arc intersections (bridge under, arc
   over) → traverse the closed curve → number edges (bottommost = 1, CCW).
3. **encode** — PD code `X[a,b,c,d]` (KnotTheory convention, sorted by `a`) and a
   signed Gauss code.

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # needs `pdftoppm` (poppler) for PDFs
```

## Command line

Input is an image (PNG / JPG) or a **single-page** PDF (multi-page PDFs are rejected).

```bash
./.venv/bin/python -m knotreader sample_knot.jpg --excl 8 --out out
```

Prints the Gauss + PD codes per detected diagram and writes `*_overlay.png` to `out/`.

## Web UI

```bash
./.venv/bin/python webapp/app.py        # http://127.0.0.1:5001
```

Upload an image, adjust **sensitivity** (`excl`: lower detects tighter crossings),
and get the processed image, the crossing/edge overlay, and the Gauss + PD codes.

## License

MIT — see [LICENSE](LICENSE).
