import asyncio
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from nail_try_on.config import (
    IMAGE_QUALITY,
    MAX_CAPTURE_DIM,
    MAX_DETECTION_DIM,
    MAX_SEND_FPS,
    NAIL_ALPHA,
    _hex_to_rgb,
    get_static_dir,
)
from nail_try_on.services.painting import process_frame_with_hand_status


app = FastAPI()
app.mount("/static", StaticFiles(directory=str(get_static_dir())), name="static")


@app.get("/")
async def get_index():
    with open(get_static_dir() / "index.html", "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("{{MAX_CAPTURE_DIM}}", str(MAX_CAPTURE_DIM))
    html = html.replace("{{MAX_SEND_FPS}}", str(MAX_SEND_FPS))
    html = html.replace("{{NAIL_ALPHA}}", str(NAIL_ALPHA))
    html = html.replace("{{IMAGE_QUALITY}}", str(IMAGE_QUALITY))
    return HTMLResponse(html)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    color: str = "FF0000",
    opacity: str = str(NAIL_ALPHA),
):
    await websocket.accept()
    from nail_try_on.config import MAX_PROCESS_FPS, NO_HAND_COOLDOWN, ROBOFLOW_MAX_DIM

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
                        import json

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
                    process_frame_with_hand_status,
                    data,
                    MAX_DETECTION_DIM,
                    ROBOFLOW_MAX_DIM,
                    current_color,
                    current_opacity,
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


@app.post("/paint")
async def paint_endpoint(
    file: UploadFile = File(...),
    color: str = Query("FF0000"),
    opacity: str = Query(str(NAIL_ALPHA)),
):
    """Process an uploaded image and return the painted result."""
    from nail_try_on.config import NAIL_BLUR, ROBOFLOW_MAX_DIM

    image_bytes = await file.read()
    processed, _ = await asyncio.to_thread(
        process_frame_with_hand_status,
        image_bytes,
        MAX_DETECTION_DIM,
        ROBOFLOW_MAX_DIM,
        _hex_to_rgb(color),
        float(opacity),
        NAIL_BLUR * 4,
        live_preview=False,
    )
    return Response(content=processed, media_type="image/jpeg")
