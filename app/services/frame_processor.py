from PIL import Image, ImageOps
from io import BytesIO

from app.config import (
    FRAME_SKIPPED_BLUR_THRESHOLD,
    RED,
    NAIL_ALPHA,
    NAIL_BLUR,
)
from app.services.hand_detector import detect_hands, is_blur
from app.services.nail_detector import detect_nails, filter_nails_by_hands
from app.services.nail_painter import paint_nails


def process_frame_with_hand_status(
    image_bytes: bytes,
    max_dim: int,
    roboflow_max_dim: int,
    color: tuple = RED,
    alpha: float = NAIL_ALPHA,
    blur: int = NAIL_BLUR,
):
    """Process a frame and return (processed_bytes, hands_found_bool, reason)."""
    try:
        # Layer 1: Blur check
        if is_blur(image_bytes, threshold=FRAME_SKIPPED_BLUR_THRESHOLD):
            return image_bytes, False, "blur"

        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            image = img.convert("RGB")
            width, height = image.size

        # Layer 2: Hand detection
        hands_data = _detect_hands(image_bytes, max_dim=max_dim, preloaded_image=image)

        if not hands_data:
            return image_bytes, False, "no_hands"

        # Layer 3: Nail detection + filtering
        nails = detect_nails(image_bytes, max_dim=roboflow_max_dim)
        raw_predictions = nails.get("predictions", [])

        if not raw_predictions:
            return image_bytes, False, "no_nail_detections"

        predictions = filter_nails_by_hands(nails, hands_data, width, height)

        if not predictions:
            # FALLBACK: relaxed filter — keep any nail containing a fingertip
            predictions = _relaxed_nail_filter(nails, hands_data, width, height)

        if not predictions:
            return image_bytes, False, "no_nails_near_fingertips"

        # Layer 4: Paint
        result = paint_nails(
            image_bytes, predictions, color=color, alpha=alpha, blur=blur, preloaded_image=image
        )
        if isinstance(result, bytes):
            return result, True, "painted"

    except Exception as e:
        print("Error processing frame:", str(e))
        return image_bytes, False, f"error:{e}"

    return image_bytes, False, "unknown"


def _relaxed_nail_filter(nails_result, hands_data, width, height):
    """Fallback: keep any nail prediction that contains at least one fingertip
    anywhere inside the polygon (not just near the center)."""
    if not hands_data:
        return []

    fingertips_px = []
    for hand in hands_data:
        for tip in hand.get("fingertips", []):
            fingertips_px.append((tip["x"] * width, tip["y"] * height))

    relaxed = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]

        for fx, fy in fingertips_px:
            if cv2_point_in_polygon(fx, fy, polygon):
                relaxed.append(pred)
                break

    return relaxed


def cv2_point_in_polygon(x, y, polygon):
    """Use cv2.pointPolygonTest for robust inside-polygon check."""
    import cv2
    import numpy as np
    poly_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly_np, (float(x), float(y)), False) >= 0
