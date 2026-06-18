"""Web UI for knot-reader: upload an image/PDF -> overlay + Gauss/PD codes."""

import base64
import sys
import tempfile
from pathlib import Path

# allow `python webapp/app.py` from anywhere by putting the repo root on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from flask import Flask, jsonify, render_template, request

from knotreader.split import diagrams_from_path
from knotreader.trace import diagram_to_pd, validate_pd, sort_pd, gauss_code
from knotreader.overlay import draw_overlay

app = Flask(__name__)


def png_data_uri(bgr_or_gray):
    ok, buf = cv2.imencode(".png", bgr_or_gray)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/process")
def process():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="No file uploaded."), 400
    try:
        excl = float(request.form.get("excl", 8.0))
    except ValueError:
        excl = 8.0

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        path = tmp.name

    try:
        binary, diagrams = diagrams_from_path(path)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"Could not read image: {e}"), 400

    results = []
    for d in diagrams:
        try:
            PD, info = diagram_to_pd(d.binary, excl=excl)
            PD = sort_pd(PD)
            overlay = draw_overlay(255 - d.binary, info)
            results.append(dict(
                name=d.name,
                crossings=len(PD),
                valid=bool(validate_pd(PD)),
                overlay=png_data_uri(overlay),
                gauss=" ".join(gauss_code(info)),
                pd="[" + ",".join("[" + ",".join(map(str, q)) + "]" for q in PD) + "]",
            ))
        except Exception as e:  # noqa: BLE001
            results.append(dict(name=d.name, error=str(e)))

    return jsonify(
        processed=png_data_uri(255 - binary),
        count=len(diagrams),
        excl=excl,
        results=results,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
