# knot-reader — project guide

Reads a knot-diagram image (PNG/JPG or single-page PDF) and outputs its **Gauss code**
and **PD code**, plus a verification overlay (circled crossings, numbered edges).
Pure computer-vision + geometry — no ML.

## Setup (do this on each machine)

- Use **Python 3.12** (3.14 lacks some OpenCV/scikit-image wheels).
- `python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt`
- System dependency: **poppler** for PDF input (`pdftoppm`, `pdfinfo`). macOS: `brew install poppler`.
- `.venv/` and input data are **gitignored**, so recreate the venv and copy your own
  images into `sample/` on each machine (the diagrams we work on are confidential and
  are not committed).

## Run

- CLI: `./.venv/bin/python -m knotreader sample/your_diagram.png --excl 8 --out out`
- Web UI: `./.venv/bin/python webapp/app.py` → http://127.0.0.1:5001
  (Flask debug; saving any `.py` auto-restarts the server, template edits show on refresh).

## Architecture (`knotreader/`)

- **split.py** — input → cropped diagrams. `load_gray` (rasterize PDF at ~300 dpi /
  read image; composite RGBA onto white; reject multi-page PDFs). `find_diagrams` crops
  each knot as a large connected ink component. `diagrams_from_path` is the main entry;
  it upscales small crops **on the grayscale** before thresholding (binary upscaling
  would fuse adjacent strands).
- **trace.py** — the core (one cropped binary → PD). Pipeline:
  1. `skeleton_arcs`: drop dashes (small components), skeletonize, order each component
     into an (x,y) polyline = an **arc** (a solid/over strand, broken at under-passages).
  2. `pair_bridges`: join each broken end to its continuation by **velocity continuity**
     (Hermite curve) → a **bridge** (an under-passage).
  3. `find_crossings`: crossings = **bridge × arc** intersections (bridge = under, arc =
     over). Own connecting arcs are NOT skipped (a strand can self-cross at an R1 kink);
     coincident same-(bridge,arc) hits are merged.
  4. `traverse` + `build_encounters`: walk the single closed curve through arcs/bridges.
  5. Orientation is normalized **once** in `diagram_to_pd`: if the walk is clockwise,
     reverse the *whole* traversal (steps + direction flags) and rebuild. **Never** reverse
     just one downstream view — that was a real bug (overlay/PD disagreed).
  6. `label_edges` (bottommost edge = 1), `assemble_pd`, `gauss_code`, `sort_pd`,
     `validate_pd`.
- **overlay.py** — `draw_overlay`: bridges (orange), crossing circles (red), edge labels.
- **webapp/** — Flask app + single-page Tailwind UI (drag-drop, sensitivity slider = `excl`).

## Conventions

- **Drawing**: the under-strand is **broken** (gap, often short dashes); the over-strand is
  **solid-continuous**. This is how over/under is recovered.
- **PD code**: `X[a,b,c,d]`, KnotTheory convention — `a` = incoming under-strand, then
  `b,c,d` counterclockwise, so the under-strand runs `a→c`. Output is **sorted by `a`** and
  emitted as a plain double array `[[a,b,c,d],...]`.
- **Edges**: numbered with the **bottommost edge = 1**, traversing **counterclockwise**.
  A diagram with N crossings has 2N edges. An R1 kink appears as a repeated label, e.g.
  `[44,45,45,46]`.
- `validate_pd`: every edge label must appear exactly twice (sanity check).
- `excl` (px) is the user-facing "sensitivity": lower detects tighter crossings.

## Known limits / gotchas

- Tracing params are tuned for **~300-dpi** rasterization (knot ≳ 700–1000 px on its long
  side). Lower-res inputs are fragile — pairing can fail and raises a clear error rather
  than emitting a wrong code. Prefer high-res images.
- Supports a **single knot**, and any number of **spatially-separate** knots in one image.
  **Non-split links** (connected, multi-component) are NOT yet supported — `traverse`
  assumes one closed cycle. (Planned: restart from unvisited arcs and continue numbering.)
- Multi-page PDFs are rejected by design (give a single page or an image).
- Validated end-to-end on the exotic-disk pair K₁ / K₁' → 44 crossings each.

## Working style

- Keep git commits **minimal and granular** (one logical change each).
- Do **not** commit input images (confidential; `sample/` is gitignored).
- Deferred work: non-split link support; SnapPy/Sage invariant cross-check (load the PD,
  confirm one component, compare invariants of the two diagrams).
