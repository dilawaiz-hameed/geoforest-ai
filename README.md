# GeoForest AI

Individual tree detection, counting and size classification from aerial and
satellite imagery, with landscape-scale forest-cover mapping.

Final Year Project, BS Data Science, FAST-NUCES Islamabad, in collaboration
with RESOLVE (ASCEND Tower, NSTP, NUST).

This repository is the **core module**: tile a scene, detect crowns in each
tile, merge detections across tile boundaries, and report a count. It is
deployed as a live interface and is the artifact behind the FYP-1 mid
evaluation.

## Why the merge step is the module

A tree sitting in the overlap between two tiles is reported by both of them.
Add the per-tile numbers and the count is wrong by however much the tiles
overlap. So the pipeline reports three numbers, not one:

| Field | Meaning |
| --- | --- |
| `raw_count` | Detections summed across tiles, duplicates included |
| `duplicates_removed` | Cross-tile reports collapsed by non-maximum suppression |
| `count` | Trees — what the user is told |

They always reconcile: `count + duplicates_removed == raw_count`. Raising the
overlap raises `raw_count` and `duplicates_removed` and leaves `count` alone.
With five trees planted in a synthetic scene:

| Overlap | Tiles | `raw_count` | Removed | `count` |
| --- | --- | --- | --- | --- |
| 0.00 | 6 | 7 | 2 | 5 |
| 0.10 | 9 | 10 | 5 | 5 |
| 0.25 | 9 | 10 | 5 | 5 |
| 0.50 | 12 | 12 | 7 | 5 |

That invariance is asserted in `tests/test_pipeline.py`, not just observed.

## Layout

```
src/geoforest/     core module, pure NumPy — no ML dependency
  geometry.py        IoU, non-maximum suppression, coordinate shifting
  tiling.py          overlapping tile grid with a clamped final origin
  detector.py        Detector protocol, stub, DeepForest adapter
  pipeline.py        tile -> detect -> shift -> merge -> count
api/main.py        FastAPI service (/health, /detect)
app/streamlit_app.py  operator interface; talks to the API over HTTP
tests/             94 tests, no network, no model weights
scripts/           reproduces the figures quoted above
```

### The detector is swappable on purpose

`geoforest.detector` defines a `Detector` protocol — `name`,
`is_real_model`, `predict(tile) -> (N, 5)`. Two implementations satisfy it:

- `StubDetector` — deterministic synthetic boxes seeded from the tile's own
  bytes. No weights, no torch, runs in CI in milliseconds. It exists so the
  pipeline, the API, the interface and the deployment can all be tested and
  demonstrated independently of the model.
- `DeepForestDetector` — the pretrained RetinaNet crown detector, imported
  lazily so the stub path never pays for torch.

Select with `GEOFOREST_DETECTOR=stub|deepforest`. **Every response and every
screen carries `is_real_model`**, so a stub count is never mistaken for a
measurement.

## Running it

```bash
pip install -r requirements-dev.txt
export PYTHONPATH=src

pytest                                   # the suite
uvicorn api.main:app --reload            # API on :8000, docs at /docs
GEOFOREST_API_URL=http://127.0.0.1:8000 \
  streamlit run app/streamlit_app.py     # interface on :8501
```

Both services in one container, as deployed:

```bash
docker build -t geoforest .
docker run --rm -p 7860:7860 geoforest   # interface on :7860
```

With the real model (installs CPU torch, ~2 GB):

```bash
docker build --build-arg INSTALL_ML=true -t geoforest:ml .
docker run --rm -p 7860:7860 -e GEOFOREST_DETECTOR=deepforest geoforest:ml
```

## API

`GET /health` → service status, active detector, `is_real_model`.

`POST /detect` → multipart image upload, with optional `tile_size`,
`overlap` and `iou_threshold` query parameters. Returns the three counts,
run metadata and one record per detection. Rejects non-images (415), files
over 32 MB or edges over 4096 px (413), and invalid parameters (422).

## CI/CD

`main` is protected: no direct pushes, no self-merges, CI green required.

| Workflow | Trigger | Does |
| --- | --- | --- |
| `ci.yml` | PR and push to `main`/`develop` | ruff lint and format; pytest on Python 3.10/3.11/3.12 with a 90% coverage floor on `src/geoforest`; builds the Docker image, starts it, and runs a real `/detect` request against the running container |
| `cd.yml` | push to `main` | triggers the Render deploy, then polls the live URL until it serves — a deploy that starts but never comes back up fails the job |

The image build and container smoke test are in CI, not only CD, because
"it works on my machine" and "the deployed thing answers" are different
claims and the second one is what the evaluation sees.

Deployment needs a repository secret `RENDER_DEPLOY_HOOK` and a variable
`SERVICE_URL`. The deployed service runs the same Dockerfile CI tested, on
Render's free tier, which sleeps when idle — the first request after a quiet
spell takes about a minute to wake.

## Data

The project's partner imagery is not yet available, so development uses open
sources — a fallback declared in the proposal defence, not an improvisation:

- **Maxar Open Data** — 30–50 cm VHR over Pakistan (2022 floods), free on AWS
- **NEON Tree Crowns** — annotated crown benchmark, for precision and recall
- **Sentinel-2** via Google Earth Engine — 10 m, landscape-scale mapping
- **Western Himalaya canopy height** — public GEE community asset, for context

Imagery is not committed; `data/` is ignored.

## Status

Core module, API, interface, test suite and the CI/CD path are in place and
running on the stub detector. Next: the DeepForest adapter against real
tiles, then precision and recall on the NEON annotations.
