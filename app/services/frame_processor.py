from PIL import Image, ImageOps
from io import BytesIO

from app.config import (
    MAX_DETECTION_DIM,
    ROBOFLOW_MAX_DIM,
    FRAME_SKIPPED_BLUR_THRESHOLD,
    RED,
    NAIL_ALPHA,
    NAIL_BLUR,
)
from app.services.hand_detector import _detect_hands, _is_blur
from app.services.nail_detector import _detect_nails, _filter_nails_by_hands
from app.services.nail_painter import _paint_nails


def _process_frame_with_hand_status(image_bytes: bytes, max_dim: int, roboflow_max_dim: int, color: tuple = RED, alpha: float = NAIL_ALPHA, blur=NAIL_BLUR):
    """Process a frame and return (processed_bytes, hands_found_bool)."""
    try:
        # Layer 1: Check if the image is blurry. If it is, return the original image.
        if _is_blur(image_bytes, threshold=FRAME_SKIPPED_BLUR_THRESHOLD):
            print("Frame is too blurry, skipping processing.")  
            return image_bytes, False

        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            image = img.convert("RGB")
            width, height = image.size
        
        # Layer 2: Detect hands. If hands are found, proceed to detect nails and paint them. Otherwise, return the original image.
        hands_data = _detect_hands(image_bytes, max_dim=max_dim, preloaded_image=image)

        if hands_data and len(hands_data) > 0:
            print(f"Detected {len(hands_data)} hands.")
             # Layer 3: Detect nails and filter by hands.
            nails = _detect_nails(image_bytes, max_dim=roboflow_max_dim)

            predictions = _filter_nails_by_hands(nails, hands_data, width, height)
            if predictions and len(predictions) > 0:
                print(f"Detected {len(predictions)} nail regions after filtering by hands.")
                result = _paint_nails(image_bytes, predictions, color=color, alpha=alpha, blur=blur, preloaded_image=image)
                if isinstance(result, bytes):
                    return result, True
    except Exception as e:
        print("Error processing frame:", str(e))
    return image_bytes, False
