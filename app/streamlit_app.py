"""GeoForest AI — operator interface for the detection and counting module.

Thin client: everything numeric happens in the FastAPI service, so the
number on screen is the same number the API returns and the report quotes.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("GEOFOREST_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = 300

st.set_page_config(page_title="GeoForest AI", page_icon="🌲", layout="wide")


def api_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    pen = ImageDraw.Draw(canvas)
    for box in detections:
        pen.rectangle(
            [box["x0"], box["y0"], box["x1"], box["y1"]],
            outline=(255, 215, 0),
            width=2,
        )
    return canvas


st.title("GeoForest AI")
st.caption(
    "Individual tree detection and counting from aerial and satellite imagery. "
    "FYP-1 core module."
)

health = api_health()

with st.sidebar:
    st.header("Service")
    if health is None:
        st.error("Detection service unreachable.")
        st.caption(f"Tried `{API_URL}/health`")
    else:
        st.success("Detection service online")
        st.metric("Model", health["detector"])
        if not health["is_real_model"]:
            st.warning(
                "Running the **stub detector**: boxes are synthetic placeholders "
                "that exercise the pipeline. Counts are not measurements."
            )
        st.caption(f"API version {health['version']}")

    st.header("Parameters")
    tile_size = st.select_slider("Tile size (px)", [200, 256, 320, 400, 512], value=400)
    overlap = st.slider("Tile overlap", 0.0, 0.5, 0.25, 0.05)
    iou_threshold = st.slider("Merge IoU threshold", 0.1, 0.7, 0.3, 0.05)
    st.caption(
        "Overlap stops crowns being cut at tile edges. The merge threshold then "
        "collapses the duplicate reports that overlap creates."
    )

upload = st.file_uploader(
    "Scene (PNG, JPG or TIF, up to 4096 px per edge)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if upload is None:
    st.info(
        "Upload a scene to begin. Public sources used in this project: "
        "Maxar Open Data (Pakistan, 30–50 cm), NEON Tree Crowns (annotated "
        "benchmark), Sentinel-2 via Google Earth Engine (10 m)."
    )
    st.stop()

payload = upload.getvalue()
scene = Image.open(io.BytesIO(payload))
st.caption(f"{upload.name} — {scene.size[0]} × {scene.size[1]} px")

if not st.button("Detect and count", type="primary", disabled=health is None):
    st.image(scene, caption="Input scene", use_container_width=True)
    st.stop()

with st.spinner("Tiling, detecting, de-duplicating…"):
    try:
        response = requests.post(
            f"{API_URL}/detect",
            params={
                "tile_size": tile_size,
                "overlap": overlap,
                "iou_threshold": iou_threshold,
            },
            files={"file": (upload.name, payload, upload.type or "image/png")},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()

if response.status_code != 200:
    detail = response.json().get("detail", response.text)
    st.error(f"Service returned {response.status_code}: {detail}")
    st.stop()

result = response.json()

a, b, c, d = st.columns(4)
a.metric("Trees counted", result["count"])
b.metric("Raw detections", result["raw_count"])
c.metric("Duplicates merged", result["duplicates_removed"])
d.metric("Tiles processed", result["tile_count"])

st.caption(
    f"{result['raw_count']} per-tile detections → {result['count']} trees after "
    f"cross-tile merging, in {result['elapsed_s']} s on "
    f"`{result['detector_name']}`."
)
if not result["is_real_model"]:
    st.warning("Stub detector: this count is a pipeline check, not a measurement.")

left, right = st.columns([3, 2])
with left:
    st.image(
        draw_boxes(scene, result["detections"]),
        caption=f"{result['count']} crowns",
        use_container_width=True,
    )
with right:
    table = pd.DataFrame(result["detections"])
    st.dataframe(table, height=420, use_container_width=True)
    if not table.empty:
        st.download_button(
            "Download detections (CSV)",
            table.to_csv(index=False).encode(),
            file_name=f"{upload.name.rsplit('.', 1)[0]}_detections.csv",
            mime="text/csv",
        )
    with st.expander("Run metadata"):
        st.json({k: v for k, v in result.items() if k != "detections"})
