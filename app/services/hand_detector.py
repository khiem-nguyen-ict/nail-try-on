import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from io import BytesIO
from typing import Union

from app.config import (
    MAX_DETECTION_DIM,
    ROBOFLOW_MAX_DIM,
    FRAME_SKIPPED_BLUR_THRESHOLD,
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.8
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]

# Map each fingertip ID to its corresponding DIP (or IP for thumb) landmark ID
FINGERTIP_DIP_MAP = {
    4: 3,   # Thumb: TIP -> IP
    8: 7,   # Index: TIP -> DIP
    12: 11, # Middle: TIP -> DIP
    16: 15, # Ring: TIP -> DIP
    20: 19, # Pinky: TIP -> DIP
}


def _is_blur(image_bytes: bytes, threshold: float = FRAME_SKIPPED_BLUR_THRESHOLD) -> bool:
    """Return True if the image is too blurry, False otherwise.

    Uses the variance of the Laplacian method. A sharp image has high
    frequency details and thus a high Laplacian variance. A blurry
    image has a low variance.

    Args:
        image_bytes: JPEG image bytes.
        threshold: Laplacian variance threshold. Images with variance
            below this value are considered blurry.

    Returns:
        True if the image is blurry, False otherwise.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return True

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    except Exception:
        return True


def detect_hands(image_source: Union[str, bytes], max_dim: int = 0, preloaded_image: Union[Image.Image, None] = None):
    """Detect hands in an image and return finger tips in JSON format.

    Args:
        image_source: Path to the image file or JPEG bytes.
        max_dim: Maximum dimension for the image used for detection.
            Smaller values are faster but may reduce accuracy.
            When 0 (default), no downscaling is applied.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.

    Returns:
        A JSON string representing a list of detected hands.
    """
    if preloaded_image is not None:
        image = preloaded_image
    elif isinstance(image_source, bytes):
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_source)))
    else:
        image = ImageOps.exif_transpose(Image.open(image_source))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Downscale for faster detection
    detection_image = image
    if max_dim > 0:
        w, h = image.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            detection_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    image_np = np.array(detection_image)

    results = hands_detector.process(image_np)
    
    output = []
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_entry = {
                "hand_index": idx,
                "handedness": None,
                "fingertips": []
            }

            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_entry["handedness"] = results.multi_handedness[idx].classification[0].label

            for lm_id in FINGERTIP_IDS:
                lm = hand_landmarks.landmark[lm_id]
                dip_id = FINGERTIP_DIP_MAP[lm_id]
                dip_lm = hand_landmarks.landmark[dip_id]

                # Angle of the fingertip segment (DIP -> TIP) in degrees
                dx = lm.x - dip_lm.x
                dy = lm.y - dip_lm.y
                angle_deg = round(np.degrees(np.arctan2(dy, dx)), 2)

                hand_entry["fingertips"].append({
                    "landmark_id": lm_id,
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "a": angle_deg
                })

            output.append(hand_entry)

    return output
