import base64
import json
import urllib.parse
import urllib.request
from io import BytesIO
from typing import Union

from PIL import Image, ImageOps

from nail_try_on.config import ROBOFLOW_PARAMS, ROBOFLOW_URL


def detect_nails(image_source: Union[str, bytes], max_dim: int = 0) -> dict:
    """Send image to the RoBoFlow API and return the inference result.

    Args:
        image_source: Path to the source image file or JPEG bytes.
        max_dim: Optional maximum dimension for downscaling before sending
            to the API. When 0 (default), no downscaling is applied.
            Downscaling reduces API latency and response size.

    Returns:
        Parsed JSON response from the RoBoFlow API.
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
                resized = src.convert("RGB").resize(
                    (new_w, new_h), Image.Resampling.LANCZOS
                )
            buf = BytesIO()
            resized.save(buf, format="JPEG", quality=80)
            send_bytes = buf.getvalue()

    base64_encoded = base64.b64encode(send_bytes)

    query_string = urllib.parse.urlencode(ROBOFLOW_PARAMS)
    full_url = f"{ROBOFLOW_URL}?{query_string}"

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
