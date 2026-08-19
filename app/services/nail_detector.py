import base64
import json
import math
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Union
from PIL import Image, ImageOps

import numpy as np

from app.config import (
    PARAMS,
    URL,
)


def detect_nails(image_source: Union[str, bytes], max_dim: int = 0):
    """Send image to the RoBoFlow API and return the inference result.

    Args:
        image_source: Path to the source image file or JPEG bytes.
        max_dim: Optional maximum dimension for downscaling before sending
            to the API. When 0 (default), no downscaling is applied.
            Downscaling reduces API latency and response size.
    """
    if isinstance(image_source, bytes):
        image_bytes = image_source
    else:
        with open(image_source, "rb") as image_file:
            image_bytes = image_file.read()

    send_bytes = image_bytes
    scale = 1.0
    if max_dim > 0:
        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            w, h = img.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as src:
                resized = src.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
            buf = BytesIO()
            resized.save(buf, format="JPEG", quality=80)
            send_bytes = buf.getvalue()

    base64_encoded = base64.b64encode(send_bytes)

    query_string = urllib.parse.urlencode(PARAMS)
    full_url = f"{URL}?{query_string}"

    req = urllib.request.Request(
        full_url,
        data=base64_encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Scale polygon coordinates back to original image space if downscaled
    if scale < 1.0:
        inv_scale = 1.0 / scale
        for pred in result.get("predictions", []):
            for point in pred.get("points", []):
                point["x"] = float(point["x"]) * inv_scale
                point["y"] = float(point["y"]) * inv_scale

    return result


def _point_in_polygon(x, y, polygon):
    """Ray-casting test: return True if (x, y) lies inside the polygon."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px, py) to segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _point_to_polygon_distance(x, y, polygon):
    """Distance from a point to a polygon.

    Returns 0.0 when the point is inside the polygon, otherwise the
    shortest distance to the polygon outline. This is robust to the case
    where a fingertip sits just outside the nail polygon (e.g. at the free
    edge of the nail) and therefore is never strictly "contained".
    """
    if not polygon:
        return float("inf")
    if _point_in_polygon(x, y, polygon):
        return 0.0
    min_dist = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = _point_to_segment_distance(x, y, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist


def _nail_orientation(polygon):
    """Orientation (degrees) of a nail's long axis, via PCA of its vertices.

    Used as a fallback when a nail cannot be paired with a fingertip, so the
    nail is still returned with a sensible rotation instead of being dropped.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    # Principal axis = eigenvector of the largest eigenvalue of the covariance.
    w, v = np.linalg.eigh(cov)
    principal = v[:, int(np.argmax(w))]
    return float(math.degrees(math.atan2(principal[1], principal[0])))


def filter_nails_by_hands(nails_result, hands_data, width, height):
    """Annotate every detected nail with the finger it belongs to.

    Matching strategy
    -----------------
    For every (nail, fingertip) pair we compute the distance from the
    fingertip to the nail polygon (0 when the tip is inside the polygon,
    otherwise the shortest distance to its outline). We then solve a
    one-to-one assignment: each fingertip is claimed by at most one nail and
    each nail gets its nearest *available* fingertip.

    This removes the old ``SPACE_DETECTION_THRESHOLD`` dilemma:
      * Wrong-finger matches are prevented by the exclusive one-to-one
        assignment -- a fingertip can no longer be shared by several nails.
      * No nail is ever dropped. A nail paired with a fingertip inherits that
        finger's angle/z/landmark (the most accurate orientation). A nail that
        cannot be paired (e.g. more nails than fingertips, or a stray
        detection) falls back to the orientation derived from its own polygon
        geometry, so it is still painted rather than missed.

    Returns one entry per detected nail prediction.
    """
    predictions = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        predictions.append(pred)

    if not predictions:
        return []

    polygons = [
        [(float(p["x"]), float(p["y"])) for p in pred.get("points", [])]
        for pred in predictions
    ]

    fingertips_px = []
    for hand in hands_data:
        for tip in hand.get("fingertips", []):
            fingertips_px.append({
                "x": tip["x"] * width,
                "y": tip["y"] * height,
                "a": tip["a"],
                "z": tip.get("z"),
                "a3d": tip.get("a3d"),
                "landmark_id": tip.get("landmark_id"),
            })

    # One-to-one assignment of fingertips to nails (nearest available wins).
    # Process pairs from nearest to farthest; because pairs are sorted by
    # distance, the first time a nail appears it is with its closest
    # fingertip and the first time a fingertip appears it is with its closest
    # nail. Skipping already-used entries keeps the assignment exclusive.
    nail_to_ft = {}
    if fingertips_px:
        pairs = []
        for ni, polygon in enumerate(polygons):
            for fi, ft in enumerate(fingertips_px):
                dist = _point_to_polygon_distance(ft["x"], ft["y"], polygon)
                pairs.append((dist, ni, fi))

        used_ft = set()
        for _dist, ni, fi in sorted(pairs, key=lambda p: p[0]):
            if ni in nail_to_ft or fi in used_ft:
                continue
            nail_to_ft[ni] = fi
            used_ft.add(fi)

    filtered = []
    for ni, pred in enumerate(predictions):
        fi = nail_to_ft.get(ni)
        pred_copy = dict(pred)

        if fi is not None:
            ft = fingertips_px[fi]
            pred_copy["angle"] = ft["a"]
            if ft.get("z") is not None:
                pred_copy["z"] = ft["z"]
            if ft.get("a3d") is not None:
                pred_copy["a3d"] = ft["a3d"]
            if ft.get("landmark_id") is not None:
                pred_copy["landmark_id"] = ft["landmark_id"]
        else:
            # No nearby finger: derive orientation from the nail shape itself.
            pred_copy["angle"] = _nail_orientation(polygons[ni])

        rounded_points = []
        for p in pred_copy.get("points", []):
            rounded_point = {
                "x": round(float(p.get("x", 0)), 12),
                "y": round(float(p.get("y", 0)), 12),
                "z": round(float(p.get("z", 0)), 12),
            }
            for k, v in p.items():
                if k not in ("x", "y", "z"):
                    rounded_point[k] = v
            rounded_points.append(rounded_point)
        pred_copy["points"] = rounded_points

        for key in ("mask_format", "confidence", "class", "class_id", "detection_id"):
            pred_copy.pop(key, None)

        filtered.append(pred_copy)

    return filtered
