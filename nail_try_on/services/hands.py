import math
from io import BytesIO
from typing import List, Optional, Union

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

from nail_try_on.config import MAX_DETECTION_DIM

# MediaPipe hands detector (singleton)
_mp_hands = mp.solutions.hands
_hands_detector = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.8,
)

# Mapping each fingertip landmark ID to its corresponding joint ID
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_JOINT_MAP = {
    4: 3,  # Thumb:  THUMB_IP (3)   -> THUMB_TIP (4)
    8: 7,  # Index:  INDEX_FINGER_DIP (7) -> INDEX_FINGER_TIP (8)
    12: 11,  # Middle: MIDDLE_FINGER_DIP (11) -> MIDDLE_FINGER_TIP (12)
    16: 15,  # Ring:   RING_FINGER_DIP (15) -> RING_FINGER_TIP (16)
    20: 19,  # Pinky:  PINKY_DIP (19) -> PINKY_TIP (20)
}


def _detect_hands(
    image_source: Union[str, bytes],
    max_dim: int = 0,
    preloaded_image: Optional[Image.Image] = None,
) -> List[dict]:
    """Detect hands in an image and return fingertips with their rotation angles.

    Args:
        image_source: Path to the image file or JPEG bytes.
        max_dim: Maximum dimension for the image used for detection.
            Smaller values are faster but may reduce accuracy.
            When 0 (default), no downscaling is applied.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.

    Returns:
        A list of dicts representing detected hands and fingertip orientations.
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

    results = _hands_detector.process(image_np)

    output = []
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_entry = {
                "hand_index": idx,
                "handedness": None,
                "fingertips": [],
            }

            if results.multi_handedness and idx < len(results.multi_handedness):
                hand_entry["handedness"] = results.multi_handedness[idx].classification[0].label

            for lm_id in FINGERTIP_IDS:
                tip_lm = hand_landmarks.landmark[lm_id]

                # Retrieve the corresponding joint landmark
                joint_id = FINGER_JOINT_MAP.get(lm_id, lm_id - 1)
                joint_lm = hand_landmarks.landmark[joint_id]

                # Calculate 2D direction vector (dx, dy) in normalized image space
                dx = tip_lm.x - joint_lm.x
                dy = tip_lm.y - joint_lm.y

                # Calculate angle in degrees.
                # math.atan2(dy, dx) returns angle where 0° points right (+X) and 90° points down (+Y).
                # Adding 90° sets 0° to point straight up along the finger length.
                angle_rad = math.atan2(dy, dx)
                angle_deg = math.degrees(angle_rad) + 90.0

                # Normalize angle to -180° to 180° range
                angle_deg = (angle_deg + 180.0) % 360.0 - 180.0

                hand_entry["fingertips"].append({
                    "landmark_id": lm_id,
                    "x": round(tip_lm.x, 6),
                    "y": round(tip_lm.y, 6),
                    "angle": round(angle_deg, 2),
                })
            output.append(hand_entry)

    return output
