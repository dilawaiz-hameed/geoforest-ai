#!/usr/bin/env python3
"""Select Maxar Open Data tiles over an area of interest.

The ``pakistan-flooding22`` release holds 2,549 analysis-ready tiles at
roughly 0.36 m. Most of them cover flood plain rather than canopy, so running
the crown detector over all of them would waste days of compute on imagery
with no trees in it. This script narrows the index to a geographic box before
anything is downloaded, and reports what it found.

The published index carries, for every tile, a footprint polygon in
longitude/latitude, the ground sample distance, the cloud percentage and a
direct HTTPS link to an RGB GeoTIFF. No authentication and no geospatial
library are needed: a footprint either overlaps the requested box or it does
not, and that is four comparisons.

Examples
--------
    # What is available over the Galiyat and Margalla ranges?
    python scripts/fetch_maxar.py --bbox 72.8 33.6 73.5 34.3

    # Same area, clear imagery only, and fetch it
    python scripts/fetch_maxar.py --bbox 72.8 33.6 73.5 34.3 \
        --max-clouds 20 --download data/maxar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

INDEX_URL = (
    "https://raw.githubusercontent.com/opengeos/maxar-open-data/master/"
    "datasets/pakistan-flooding22.geojson"
)
TIMEOUT = 60
CHUNK = 1 << 20  # 1 MiB


def load_index(source: str) -> dict:
    """Load the tile index from a URL or a local file.

    A local path is accepted so the script can be run and tested without
    network access, and so a downloaded index can be reused.
    """
    path = Path(source)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    response = requests.get(source, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def footprint_bounds(geometry: dict) -> tuple[float, float, float, float] | None:
    """Longitude/latitude bounds of a footprint.

    The index uses Polygon and MultiPolygon geometries. Rather than walk each
    type separately, every coordinate pair is collected and reduced, which is
    correct for both and does not care how deeply the rings are nested.
    """
    if not geometry:
        return None

    lons: list[float] = []
    lats: list[float] = []

    def walk(node) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and all(isinstance(v, (int, float)) for v in node[:2])
        ):
            lons.append(float(node[0]))
            lats.append(float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(geometry.get("coordinates"))
    if not lons:
        return None
    return min(lons), min(lats), max(lons), max(lats)


def overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True when two longitude/latitude boxes share any area.

    Written as the negation of the four ways they can miss each other, which
    is shorter and harder to get wrong than enumerating the overlap cases.
    Boxes that only touch along an edge count as overlapping, because a tile
    whose edge grazes the area of interest may still contain canopy inside it.
    """
    a_min_lon, a_min_lat, a_max_lon, a_max_lat = a
    b_min_lon, b_min_lat, b_max_lon, b_max_lat = b
    return not (
        a_max_lon < b_min_lon
        or a_min_lon > b_max_lon
        or a_max_lat < b_min_lat
        or a_min_lat > b_max_lat
    )


def select(
    index: dict, bbox: tuple[float, float, float, float], max_clouds: float | None
) -> list[dict]:
    """Tiles whose footprint meets the box, optionally filtered by cloud."""
    selected = []
    for feature in index.get("features", []):
        bounds = footprint_bounds(feature.get("geometry"))
        if bounds is None or not overlaps(bounds, bbox):
            continue

        properties = feature.get("properties", {})
        clouds = properties.get("tile:clouds_percent")
        if max_clouds is not None and clouds is not None and clouds > max_clouds:
            continue

        selected.append(
            {
                "quadkey": properties.get("quadkey", "?"),
                "date": str(properties.get("datetime", "?"))[:10],
                "gsd": properties.get("gsd"),
                "clouds": clouds,
                "url": properties.get("visual"),
            }
        )
    return selected


def download(tiles: list[dict], directory: Path) -> int:
    """Fetch each tile's GeoTIFF, skipping files already present.

    Tiles are hundreds of megabytes, so the response is streamed rather than
    held in memory, and anything already on disk is left alone. The script is
    expected to be re-run after an interrupted session.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fetched = 0

    for tile in tiles:
        url = tile["url"]
        if not url:
            print(f"  {tile['quadkey']}: no image link in the index, skipped")
            continue

        target = directory / f"{tile['quadkey']}_{tile['date']}.tif"
        if target.exists() and target.stat().st_size > 0:
            print(f"  {target.name}: already present, skipped")
            continue

        print(f"  {target.name}: downloading...", flush=True)
        partial = target.with_suffix(".tif.part")
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=CHUNK):
                        handle.write(block)
        except requests.RequestException as error:
            partial.unlink(missing_ok=True)
            print(f"    failed: {error}")
            continue

        # Rename only once the transfer completed, so an interrupted run never
        # leaves a truncated file that a later run would mistake for finished.
        partial.replace(target)
        fetched += 1

    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select Maxar Open Data tiles over an area of interest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="area of interest in degrees, e.g. --bbox 72.8 33.6 73.5 34.3",
    )
    parser.add_argument(
        "--max-clouds",
        type=float,
        default=None,
        help="discard tiles above this cloud percentage",
    )
    parser.add_argument(
        "--download",
        type=Path,
        default=None,
        metavar="DIR",
        help="fetch the matching GeoTIFFs into DIR instead of listing only",
    )
    parser.add_argument(
        "--index",
        default=INDEX_URL,
        help="tile index URL or local GeoJSON path (default: the published index)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="use at most this many matches, for a quick trial run",
    )
    args = parser.parse_args()

    min_lon, min_lat, max_lon, max_lat = args.bbox
    if min_lon >= max_lon or min_lat >= max_lat:
        print("Error: bbox must be MINLON MINLAT MAXLON MAXLAT, min before max.")
        return 2

    try:
        index = load_index(args.index)
    except (requests.RequestException, json.JSONDecodeError, OSError) as error:
        print(f"Error: could not read the tile index: {error}")
        return 2

    total = len(index.get("features", []))
    tiles = select(index, tuple(args.bbox), args.max_clouds)
    if args.limit is not None:
        tiles = tiles[: args.limit]

    if not tiles:
        print(f"No tiles match that area out of {total} in the index.")
        print("Widen the box, raise --max-clouds, or check the coordinate order.")
        return 1

    print(f"{len(tiles)} of {total} tiles match.\n")
    print(f"{'Quadkey':<16}{'Date':<12}{'GSD':>7}{'Cloud %':>9}  URL")
    print("-" * 100)
    for tile in tiles:
        gsd = f"{tile['gsd']:.2f}" if isinstance(tile["gsd"], (int, float)) else "?"
        clouds = (
            f"{tile['clouds']:.0f}" if isinstance(tile["clouds"], (int, float)) else "?"
        )
        print(
            f"{tile['quadkey']:<16}{tile['date']:<12}{gsd:>7}{clouds:>9}  {tile['url']}"
        )

    if args.download:
        print(f"\nDownloading into {args.download}")
        fetched = download(tiles, args.download)
        print(
            f"\n{fetched} new file(s); {len(tiles) - fetched} already present or skipped."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
