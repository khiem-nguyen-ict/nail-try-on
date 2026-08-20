import warnings

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated.*",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)

import asyncio
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import (
    MAX_CAPTURE_DIM,
    MAX_SEND_FPS,
    NAIL_ALPHA,
    IMAGE_QUALITY,
    ROBOFLOW_MAX_DIM,
    NAIL_BLUR,
    NO_HAND_COOLDOWN,
    MAX_PROCESS_FPS,
)
from app.utils.color import hex_to_rgb
from app.services.frame_processor import process_frame_with_hand_status
from app.services.pattern_static_painter import paint_with_pattern


def create_app() -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/patterns", StaticFiles(directory="sample-images"), name="patterns")

    @app.get("/")
    async def get_index():
        with open("static/index.html", "r", encoding="utf-8") as f:
            html = f.read()
        html = html.replace("{{MAX_CAPTURE_DIM}}", str(MAX_CAPTURE_DIM))
        html = html.replace("{{MAX_SEND_FPS}}", str(MAX_SEND_FPS))
        html = html.replace("{{NAIL_ALPHA}}", str(NAIL_ALPHA))
        html = html.replace("{{IMAGE_QUALITY}}", str(IMAGE_QUALITY))
        return HTMLResponse(html)

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str, color: str = "FF0000", opacity: str = str(NAIL_ALPHA)):
        await websocket.accept()
        last_process_time = 0.0
        min_interval = 1.0 / MAX_PROCESS_FPS
        skip_until = 0
        current_color = hex_to_rgb(color)
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
                                    current_color = hex_to_rgb(new_color)
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

                    processed, hands_found, reason = await asyncio.to_thread(
                        process_frame_with_hand_status, data, ROBOFLOW_MAX_DIM, current_color, current_opacity
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
        image_bytes = await file.read()
        processed, _, _ = await asyncio.to_thread(
            process_frame_with_hand_status,
            image_bytes,
            ROBOFLOW_MAX_DIM,
            hex_to_rgb(color),
            float(opacity),
            NAIL_BLUR
        )
        return Response(content=processed, media_type="image/jpeg")

    @app.post("/paint_pattern")
    async def paint_pattern_endpoint(
        file: UploadFile = File(...),
        pattern: str = Query(...),
    ):
        """Overlay the selected nail pattern onto the uploaded hand image."""
        image_bytes = await file.read()
        safe_name = os.path.basename(pattern)
        pattern_path = os.path.join("sample-images", safe_name)

        if not os.path.isfile(pattern_path):
            return Response(content=image_bytes, media_type="image/jpeg", status_code=400)

        processed = await asyncio.to_thread(
            paint_with_pattern,
            image_bytes,
            pattern_path,
        )
        return Response(content=processed, media_type="image/jpeg")

    return app


app = create_app()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
