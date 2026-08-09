import asyncio
import base64
import json
import os
import time
from io import BytesIO
from typing import Union

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageOps
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("ROBOFLOW_API_KEY"):
    raise RuntimeError("ROBOFLOW_API_KEY environment variable is not set. Please set it to your RoBoFlow API key.")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# Default fill color: solid red.
RED = (255, 0, 0)
NAIL_ALPHA = float(os.getenv("NAIL_ALPHA", "0.4"))
NAIL_BLUR = int(os.getenv("NAIL_BLUR", "1"))
SPACE_DETECTION_THRESHOLD = float(os.getenv("SPACE_DETECTION_THRESHOLD", "0.1"))
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
MAX_DETECTION_DIM = int(os.getenv("MAX_DETECTION_DIM", "320"))
MAX_PROCESS_FPS = int(os.getenv("MAX_PROCESS_FPS", "20"))
NO_HAND_COOLDOWN = float(os.getenv("NO_HAND_COOLDOWN", "1.0"))
ROBOFLOW_MAX_DIM = int(os.getenv("ROBOFLOW_MAX_DIM", "1024"))

TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)


# Configuration
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": os.getenv("ROBOFLOW_API_KEY", ""), "confidence": YOLO_CONFIDENCE_THRESHOLD}

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,           
    min_detection_confidence=0.95, 
    min_tracking_confidence=0.95   
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert a hex color string (e.g. '#FF0000' or 'FF0000') to an RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return RED
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return RED


def _detect_nails(image_source: Union[str, bytes], max_dim: int = 0):
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
            resized.save(buf, format="JPEG", quality=90)
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


def _apply_color_transfer(
    bgr: np.ndarray,
    mask: np.ndarray,
    target_hsv: np.ndarray = TARGET_HSV,
    alpha: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = target_hsv[0]
    hsv[:, :, 1] = target_hsv[1]
    hsv[:, :, 2] = target_hsv[2]

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (bgr.astype(np.float32) * (1.0 - mask_float * alpha) + recolored_bgr.astype(np.float32) * mask_float * alpha).astype(np.uint8)
    return result


def _paint_nails(image_source: Union[str, bytes], regions, color=RED, alpha: float = 1.0, blur: int = 0, preloaded_image: Union[Image.Image, None] = None):
    """Paint detected nail regions on the image with ``color``.

    Args:
        image_source: Path to the source image file or JPEG bytes.
        regions: List of detected nail regions.
        color: Color to use for painting.
        alpha: Opacity of the painted regions.
        blur: Blur radius for the painted regions.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.
    """
    if preloaded_image is not None:
        image = preloaded_image.convert("RGB")
    elif isinstance(image_source, bytes):
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_source))).convert("RGB")
    else:
        image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")

    image_np = np.array(image)
    h, w = image_np.shape[:2]

    mask = Image.new("L", (w, h), 0)
    draw_mask = ImageDraw.Draw(mask)

    for pred in regions:
        points = pred.get("points")
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]
        draw_mask.polygon(polygon, fill=255)
    
    mask_np = np.array(mask)
    
    if blur > 0:
        ksize = 2 * int(blur) + 1
        mask_np = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur)
    
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    rgb_color = np.uint8([[color]])
    target_hsv = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)

    painted_bgr = _apply_color_transfer(image_bgr, mask_np, target_hsv, alpha=alpha)
    painted_rgb = cv2.cvtColor(painted_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(painted_rgb)

    if isinstance(image_source, bytes):
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    return image


import json
from io import BytesIO
from typing import Union
import cv2
import numpy as np
from PIL import Image, ImageOps

# Ensure FINGERTIP_IDS is defined (4: Thumb, 8: Index, 12: Middle, 16: Ring, 20: Pinky)
FINGERTIP_IDS = [4, 8, 12, 16, 20]

def _is_sharp(image_np: np.ndarray, threshold: float = 80.0) -> bool:
    """Layer 1: Check image sharpness using Laplacian Variance."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold

def _detect_hands(image_source: Union[str, bytes], max_dim: int = 0, preloaded_image: Union[Image.Image, None] = None):
    """Detect hands in an image and return finger tips in JSON format.

    Args:
        image_source: Path to the image file or JPEG bytes.
        max_dim: Maximum dimension for the image used for detection.
            Smaller values are faster but may reduce accuracy.
            When 0 (default), no downscaling is applied.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.

    Returns:
        A list of dicts representing detected hands and visible nail fingertips.
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

    # FILTER LAYER 1: Check image sharpness. Return empty if blurry.
    if not _is_sharp(image_np, threshold=80.0):
        return []

    results = hands_detector.process(image_np)

    output = []
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = hand_landmarks.landmark

            # Extract handedness label (Left/Right)
            handedness_label = None
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness_label = results.multi_handedness[idx].classification[0].label

            # FILTER LAYERS 2 & 3: Calculate nail orientation using 2D Cross Product
            def get_pt(lm_id):
                return np.array([landmarks[lm_id].x, landmarks[lm_id].y])

            wrist = get_pt(0)
            index_mcp = get_pt(5)
            pinky_mcp = get_pt(17)

            v_h1 = index_mcp - wrist
            v_h2 = pinky_mcp - wrist
            cross_hand = v_h1[0] * v_h2[1] - v_h1[1] * v_h2[0]

            # Check if the back of the hand is facing the camera
            is_hand_back = (cross_hand < 0) if handedness_label == "Right" else (cross_hand > 0)

            # Configuration for the 4 long fingers: (tip_id, pip_id, dip_id)
            fingers_config = [
                (8,  6,  7),   # Index
                (12, 10, 11),  # Middle
                (16, 14, 15),  # Ring
                (20, 18, 19)   # Pinky
            ]

            valid_fingertips = []

            # 1. Evaluate 4 long fingers
            for tip_id, pip_id, dip_id in fingers_config:
                pip, dip, tip = get_pt(pip_id), get_pt(dip_id), get_pt(tip_id)

                v_dip_tip = tip - dip
                v_dip_pip = pip - dip
                cross_finger = v_dip_tip[0] * v_dip_pip[1] - v_dip_pip[0] * v_dip_tip[1]

                # Finger shows nail if: the finger joint is flipped OR the back of hand faces camera
                is_finger_flipped = (cross_finger * cross_hand) < 0

                if is_finger_flipped or is_hand_back:
                    lm = landmarks[tip_id]
                    valid_fingertips.append({
                        "landmark_id": tip_id,
                        "x": round(lm.x, 6),
                        "y": round(lm.y, 6),
                        "z": round(lm.z, 6)
                    })

            # 2. Evaluate Thumb (ID 4)
            thumb_dip, thumb_tip = get_pt(3), get_pt(4)
            # Check if thumb is extended clearly and back of hand is visible
            if np.linalg.norm(thumb_tip - thumb_dip) > 0.02 and is_hand_back:
                lm = landmarks[4]
                valid_fingertips.append({
                    "landmark_id": 4,
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6)
                })

            # ONLY APPEND TO OUTPUT IF THE HAND HAS AT LEAST ONE VISIBLE NAIL FINGERTIP
            if valid_fingertips:
                # Sort fingertips back to standard landmark ID order (4, 8, 12, 16, 20)
                valid_fingertips.sort(key=lambda item: item["landmark_id"])

                hand_entry = {
                    "hand_index": idx,
                    "handedness": handedness_label,
                    "fingertips": valid_fingertips
                }
                output.append(hand_entry)

    return output


def _point_in_polygon(x, y, polygon, width, height):
    """Check if point (x, y) is close to the center of polygon."""
    if not polygon:
        return False
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    threshold = SPACE_DETECTION_THRESHOLD * (width + height) / 2
    return distance < threshold


def _filter_nails_by_hands(nails_result, hands_data, width, height):
    """Filter nail predictions to only those containing at least one fingertip."""
    if not hands_data:
        return []

    fingertips_px = []
    for hand in hands_data:
        for tip in hand.get("fingertips", []):
            fingertips_px.append((tip["x"] * width, tip["y"] * height))

    filtered = []
    for pred in nails_result.get("predictions", []):
        points = pred.get("points", [])
        if not points:
            continue
        polygon = [(float(p["x"]), float(p["y"])) for p in points]

        contained = False
        for fx, fy in fingertips_px:
            if _point_in_polygon(fx, fy, polygon, width, height):
                contained = True
                break

        if contained:
            filtered.append(pred)

    return filtered

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, color: str = "FF0000", opacity: str = str(NAIL_ALPHA)):
    await websocket.accept()
    last_process_time = 0.0
    min_interval = 1.0 / MAX_PROCESS_FPS
    skip_until = 0
    current_color = _hex_to_rgb(color)
    current_opacity = float(opacity)

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message["type"] == "websocket.receive":
                if "text" in message:
                    try:
                        payload = json.loads(message["text"])
                        if isinstance(payload, dict):
                            if payload.get("type") == "color":
                                new_color = payload.get("value", "FF0000")
                                current_color = _hex_to_rgb(new_color)
                            elif payload.get("type") == "opacity":
                                new_opacity = payload.get("value", NAIL_ALPHA)
                                current_opacity = max(0.1, min(0.8, float(new_opacity)))
                    except Exception:
                        pass
                    continue

                data = message.get("bytes")
                if data is None:
                    continue

                now = time.time()
                if now < skip_until:
                    await websocket.send_bytes(data)
                    continue
                if now - last_process_time < min_interval:
                    await websocket.send_bytes(data)
                    continue

                processed, hands_found = await asyncio.to_thread(
                    _process_frame_with_hand_status, data, MAX_DETECTION_DIM, ROBOFLOW_MAX_DIM, current_color, current_opacity
                )
                last_process_time = now

                if not hands_found:
                    skip_until = now + NO_HAND_COOLDOWN

                await websocket.send_bytes(processed)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass

def _process_frame_with_hand_status(image_bytes: bytes, max_dim: int, roboflow_max_dim: int, color: tuple = RED, alpha: float = NAIL_ALPHA):
    """Process a frame and return (processed_bytes, hands_found_bool)."""
    try:
        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            image = img.convert("RGB")
            width, height = image.size
        
        hands_data = _detect_hands(image_bytes, max_dim=max_dim, preloaded_image=image)

        if hands_data and len(hands_data) > 0:
            nails = _detect_nails(image_bytes, max_dim=roboflow_max_dim)

            predictions = _filter_nails_by_hands(nails, hands_data, width, height)
            if predictions and len(predictions) > 0:
                result = _paint_nails(image_bytes, predictions, color=color, alpha=alpha, blur=NAIL_BLUR, preloaded_image=image)
                if isinstance(result, bytes):
                    return result, True
        return image_bytes, False
    except Exception:
        return image_bytes, False


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
