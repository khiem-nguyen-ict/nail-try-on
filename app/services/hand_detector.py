import cv2
import math
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from io import BytesIO
from typing import Union

from app.config import (
    FRAME_SKIPPED_BLUR_THRESHOLD,
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.85
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


def is_blur(image_bytes: bytes, threshold: float = FRAME_SKIPPED_BLUR_THRESHOLD) -> bool:
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


def detect_hands(image_source: Union[str, bytes], preloaded_image: Union[Image.Image, None] = None, debug_save_path: Union[str, None] = None):
    """Detect hands in an image and return finger tips in JSON format.

    Args:
        image_source: Path to the image file or JPEG bytes.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.
        debug_save_path: If provided, save a debug image with all hand
            landmarks drawn as red circles to this path.

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

    image_np = np.array(image)

    results = hands_detector.process(image_np)
    
    output = []
    if results.multi_hand_landmarks:
        if debug_save_path is not None:
            debug_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            h, w = debug_image.shape[:2]
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for lm_id in FINGERTIP_IDS:
                    lm = hand_landmarks.landmark[lm_id]
                    dip_id = FINGERTIP_DIP_MAP[lm_id]
                    dip_lm = hand_landmarks.landmark[dip_id]
                
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(debug_image, (cx, cy), 15, (0, 0, 255), -1)
                    cx2, cy2 = int(dip_lm.x * w), int(dip_lm.y * h)
                    cv2.circle(debug_image, (cx2, cy2), 15, (0, 255, 0), -1)

                    # Draw dashed white line between DIP and TIP
                    dash_len = 10
                    gap_len = 10
                    dx = cx2 - cx
                    dy = cy2 - cy
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist > 0:
                        dx_u = dx / dist
                        dy_u = dy / dist
                        pos = 0
                        while pos < dist:
                            end_pos = min(pos + dash_len, dist)
                            x1 = int(cx + dx_u * pos)
                            y1 = int(cy + dy_u * pos)
                            x2 = int(cx + dx_u * end_pos)
                            y2 = int(cy + dy_u * end_pos)
                            cv2.line(debug_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            pos += dash_len + gap_len

                    cv2.putText(debug_image, f"finger: {lm_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(debug_save_path, debug_image)

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
                dz = lm.z - dip_lm.z
                angle_deg = round(np.degrees(np.arctan2(dy, dx)), 2)

                # 3-D angle between the finger (MCP -> TIP) and the camera view direction.
                # 90° = finger points toward the camera; 0° = finger lies flat on the image plane.
                plane_angle = round(np.degrees(np.arctan2(-dz, math.sqrt(dx**2 + dy**2))), 2)

                hand_entry["fingertips"].append({
                    "landmark_id": lm_id,
                    "x": round(lm.x, 12),
                    "y": round(lm.y, 12),
                    "z": round(lm.z, 12),
                    "a": angle_deg,
                    "a3d": plane_angle,
                })

            output.append(hand_entry)

    return output
