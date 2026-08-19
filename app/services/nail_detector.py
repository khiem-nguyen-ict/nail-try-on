import base64
import json
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Union
from PIL import Image, ImageOps

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


def _point_in_polygon(x, y, polygon, width, height):
    """Check if point (x, y) is inside the given polygon using ray casting."""
    if not polygon:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0) + x0):
            inside = not inside
    return inside


def filter_nails_by_hands(nails_result, hands_data, width, height):
    """Filter nail predictions to only those containing at least one fingertip."""
    if not hands_data:
        return []

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

    filtered = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]

        matched_angle = None
        matched_z = None
        matched_a3d = None
        matched_landmark_id = None
        for ft in fingertips_px:
            if _point_in_polygon(ft["x"], ft["y"], polygon, width, height):
                matched_angle = ft["a"]
                matched_z = ft.get("z")
                matched_a3d = ft.get("a3d")
                matched_landmark_id = ft.get("landmark_id")
                break

        if matched_angle is not None:
            pred_copy = dict(pred)
            pred_copy["angle"] = matched_angle
            if matched_z is not None:
                pred_copy["z"] = matched_z
            if matched_a3d is not None:
                pred_copy["a3d"] = matched_a3d
            if matched_landmark_id is not None:
                pred_copy["landmark_id"] = matched_landmark_id

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
