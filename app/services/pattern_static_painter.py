"""Static pattern painting service for nail selection feature.

Provides a function that overlays a reference nail pattern onto an uploaded
hand image using the existing detection + pattern-painting pipeline from
``experiments/static_painting.py``.
"""

from io import BytesIO

from PIL import Image, ImageOps

from app.services.hand_detector import detect_hands
from app.services.nail_detector import detect_nails, filter_nails_by_hands
from app.services.nail_pattern_painter import paint_nail_pattern
from app.utils.image import get_base_image_color_profile


def paint_with_pattern(image_bytes: bytes, pattern_path: str) -> bytes:
    """Paint the selected nail pattern onto a hand image.

    Args:
        image_bytes: JPEG bytes of the uploaded hand image.
        pattern_path: Absolute path to the reference pattern PNG/JPG.

    Returns:
        JPEG bytes of the painted image. If no hands or nails are detected,
        the original image bytes are returned unchanged.
    """
    with ImageOps.exif_transpose(Image.open(pattern_path)) as ref_image:
        sample_image = ref_image.convert("RGBA")

    base_image = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    width, height = base_image.size

    hands_data = detect_hands(image_bytes, preloaded_image=base_image)
    if not hands_data:
        return image_bytes

    nails_result = detect_nails(image_bytes)
    filtered_nails = filter_nails_by_hands(nails_result, hands_data, width, height)
    if not filtered_nails:
        return image_bytes

    filtered_nails.sort(
        key=lambda nail: (sum(float(p["z"]) for p in nail.get("points", [])), nail.get("a3d", 0))
    )

    base_color_profile = get_base_image_color_profile(base_image)

    for nail in filtered_nails:
        points = [(float(p["x"]), float(p["y"])) for p in nail.get("points", [])]
        cx = nail.get("x", 0)
        cy = nail.get("y", 0)
        angle = nail.get("angle", 0)
        a3d = nail.get("a3d", 0)
        w = nail.get("width", 0)
        h = nail.get("height", 0)
        z = nail.get("z", 0)

        paint_nail_pattern(
            base_image,
            sample_image,
            points,
            cx,
            cy,
            angle,
            w,
            h,
            z,
            a3d,
            base_color_profile,
        )

    out_buf = BytesIO()
    base_image.save(out_buf, format="JPEG", quality=90)
    return out_buf.getvalue()
