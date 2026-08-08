import asyncio
import base64
import json
import os
import sys
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
NAIL_ALPHA = float(os.getenv("NAIL_ALPHA", "0.8"))
NAIL_BLUR = int(os.getenv("NAIL_BLUR", "2"))
SPACE_DETECTION_THRESHOLD = float(os.getenv("SPACE_DETECTION_THRESHOLD", "0.1"))
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
MAX_DETECTION_DIM = int(os.getenv("MAX_DETECTION_DIM", "320"))
MAX_PROCESS_FPS = int(os.getenv("MAX_PROCESS_FPS", "20"))
NO_HAND_COOLDOWN = float(os.getenv("NO_HAND_COOLDOWN", "1.0"))
ROBOFLOW_MAX_DIM = int(os.getenv("ROBOFLOW_MAX_DIM", "640"))

TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)


# Configuration
IMAGE_PATH = "IMG_3051.JPG"
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": os.getenv("ROBOFLOW_API_KEY", ""), "confidence": YOLO_CONFIDENCE_THRESHOLD}

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.5
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]


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
            resized.save(buf, format="JPEG", quality=85)
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

    recolored_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_float = mask_3ch.astype(np.float32) / 255.0

    result = (bgr.astype(np.float32) * (1.0 - mask_float * alpha) + recolored_bgr.astype(np.float32) * mask_float * alpha).astype(np.uint8)
    return result


def paint_nails(image_source: Union[str, bytes], result, output_path=None, color=RED, alpha: float = 1.0, blur: int = 0, preloaded_image: Union[Image.Image, None] = None):
    """Paint detected nail regions on the image with ``color``.

    Args:
        image_source: Path to the source image file or JPEG bytes.
        result: Inference result dict returned by the RoBoFlow API.
        output_path: Destination path for the painted image. When
            ``None`` (the default) and bytes are provided, returns
            JPEG bytes directly instead of saving.
        preloaded_image: Optional pre-decoded PIL Image to avoid
            re-reading ``image_source``.
    """
    if preloaded_image is not None:
        image = preloaded_image.convert("RGB")
    elif isinstance(image_source, bytes):
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_source))).convert("RGB")
    else:
        image = ImageOps.exif_transpose(Image.open(image_source)).convert("RGB")

    predictions = (result or {}).get("predictions", [])
    
    if predictions:
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        mask = Image.new("L", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask)

        for pred in predictions:
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
    
    if output_path is None:
        if isinstance(image_source, bytes):
            buf = BytesIO()
            image.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        else:
            base, ext = os.path.splitext(image_source)
            output_path = f"{base}_painted{ext}"

    image.save(output_path)
    image.filename = output_path
    
    return image


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
                hand_entry["fingertips"].append({
                    "landmark_id": lm_id,
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6)
                })

            output.append(hand_entry)

    return json.dumps(output, indent=2)


def point_in_polygon(x, y, polygon, width, height):
    """Check if point (x, y) is close to the center of polygon."""
    if not polygon:
        return False
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
    threshold = SPACE_DETECTION_THRESHOLD * (width + height) / 2
    return distance < threshold


def filter_nails_by_hands(nails_result, hands_data, width, height):
    """Filter nail predictions to only those containing at least one fingertip."""
    if not hands_data:
        return {"predictions": []}

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
            if point_in_polygon(fx, fy, polygon, width, height):
                contained = True
                break

        if contained:
            filtered.append(pred)

    return {"predictions": filtered}


def process_frame(image_bytes: bytes, max_dim: int = MAX_DETECTION_DIM, roboflow_max_dim: int = ROBOFLOW_MAX_DIM) -> bytes:
    """Process a single frame using the same logic as the original main()."""
    result, _ = _process_frame_with_hand_status(image_bytes, max_dim, roboflow_max_dim)
    return result


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    last_process_time = 0.0
    min_interval = 1.0 / MAX_PROCESS_FPS
    skip_until = 0.0

    try:
        while True:
            data = await websocket.receive_bytes()
            now = time.time()
            if now < skip_until:
                await websocket.send_bytes(data)
                continue
            if now - last_process_time < min_interval:
                await websocket.send_bytes(data)
                continue

            processed, hands_found = await asyncio.to_thread(
                _process_frame_with_hand_status, data, MAX_DETECTION_DIM, ROBOFLOW_MAX_DIM
            )
            last_process_time = now

            if not hands_found:
                skip_until = now + NO_HAND_COOLDOWN

            await websocket.send_bytes(processed)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass


def _process_frame_with_hand_status(image_bytes: bytes, max_dim: int, roboflow_max_dim: int):
    """Process a frame and return (processed_bytes, hands_found_bool)."""
    try:
        with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
            image = img.convert("RGB")
            width, height = image.size

        hands = detect_hands(image_bytes, max_dim=max_dim, preloaded_image=image)
        hands_data = json.loads(hands)

        if hands_data and len(hands_data) > 0:
            nails = detect_nails(image_bytes, max_dim=roboflow_max_dim)

            filtered_nails = filter_nails_by_hands(nails, hands_data, width, height)

            predictions = filtered_nails.get("predictions", [])
            if predictions:
                result = paint_nails(image_bytes, filtered_nails, alpha=NAIL_ALPHA, blur=NAIL_BLUR, preloaded_image=image)
                if isinstance(result, bytes):
                    return result, True
            return image_bytes, True
        else:
            return image_bytes, False
    except Exception:
        return image_bytes, False


def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        try:
            hands = detect_hands(image_path)
            hands_data = json.loads(hands)

            if hands_data and len(hands_data) > 0:
                nails = detect_nails(image_path)

                with ImageOps.exif_transpose(Image.open(image_path)) as img:
                    width, height = img.size

                filtered_nails = filter_nails_by_hands(nails, hands_data, width, height)

                predictions = filtered_nails.get("predictions", [])
                if predictions:
                    paint_nails(image_path, filtered_nails, output_path=output_path, alpha=NAIL_ALPHA, blur=NAIL_BLUR)
                else:
                    pass
            else:
                pass
        except urllib.error.HTTPError as e:
            pass
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
