import asyncio
import base64
import json
import os
import sys
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

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# Default fill color: solid red.
RED = (255, 0, 0)
NAIL_ALPHA = 0.8
NAIL_BLUR = 2
TARGET_HSV = np.array([0, 255, 255], dtype=np.float32)
SPACE_DETECTION_THRESHOLD = 0.1  # 10% of average image dimension

# Configuration
IMAGE_PATH = "IMG_3051.JPG"
URL = "https://serverless.roboflow.com/thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1"
PARAMS = {"api_key": "[REDACTED]", "confidence": "0.70"}

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=4,
    model_complexity=0,
    min_detection_confidence=0.5
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]


def detect_nails(image_source: Union[str, bytes]):
    """Send image to the RoBoFlow API and return the inference result."""
    if isinstance(image_source, bytes):
        image_bytes = image_source
    else:
        with open(image_source, "rb") as image_file:
            image_bytes = image_file.read()

    base64_encoded = base64.b64encode(image_bytes)

    query_string = urllib.parse.urlencode(PARAMS)
    full_url = f"{URL}?{query_string}"

    req = urllib.request.Request(
        full_url,
        data=base64_encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


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


def paint_nails(image_source: Union[str, bytes], result, output_path=None, color=RED, alpha: float = 1.0, blur: int = 0):
    """Paint detected nail regions on the image with ``color``.

    Args:
        image_source: Path to the source image file or JPEG bytes.
        result: Inference result dict returned by the RoBoFlow API.
        output_path: Destination path for the painted image. When
            ``None`` (the default) and bytes are provided, returns
            JPEG bytes directly instead of saving.
    """
    if isinstance(image_source, bytes):
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


def detect_hands(image_source: Union[str, bytes]):
    """Detect hands in an image and return finger tips in JSON format.

    Args:
        image_source: Path to the image file or JPEG bytes.

    Returns:
        A JSON string representing a list of detected hands.
    """
    if isinstance(image_source, bytes):
        image = ImageOps.exif_transpose(Image.open(BytesIO(image_source)))
    else:
        image = ImageOps.exif_transpose(Image.open(image_source))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_np = np.array(image)

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


def process_frame(image_bytes: bytes) -> bytes:
    """Process a single frame using the same logic as the original main()."""
    try:
        hands = detect_hands(image_bytes)
        hands_data = json.loads(hands)

        if hands_data and len(hands_data) > 0:
            nails = detect_nails(image_bytes)

            with ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))) as img:
                width, height = img.size

            filtered_nails = filter_nails_by_hands(nails, hands_data, width, height)

            predictions = filtered_nails.get("predictions", [])
            if predictions:
                result = paint_nails(image_bytes, filtered_nails, alpha=NAIL_ALPHA, blur=NAIL_BLUR)
                if isinstance(result, bytes):
                    return result
            else:
                print("No nails detected that contain fingertips. Skipping nail painting.")
        else:
            print("No hands detected. Skipping nail painting.")
    except Exception as e:
        print(f"Frame processing error: {e}")
    return image_bytes


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"Session {session_id} connected")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            processed = await asyncio.to_thread(process_frame, data)
            await websocket.send_bytes(processed)
    except WebSocketDisconnect:
        print(f"Session {session_id} disconnected")
    except Exception as e:
        print(f"Session {session_id} error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


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
                    print("No nails detected that contain fingertips. Skipping nail painting.")
            else:
                print("No hands detected. Skipping nail painting.")
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.read().decode('utf-8')}")
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
