# Nail Try-On

A real-time nail polish try-on web app built with FastAPI, MediaPipe, and RoBoFlow. The frontend captures video frames via WebSocket, detects hands and nail regions, applies color transfer to the detected nails, and streams the result back — all without leaving the browser UI.

## Features

- **Real-time processing** via WebSocket (`/ws/{session_id}`)
- **Hand detection** using MediaPipe Hands
- **Nail segmentation** using a RoBoFlow RF-DETR segmentation model
- **Color transfer** to recolor detected nail regions while preserving texture
- **Performance optimizations** for shared hosting: frame rate limiting, downscaled detection, and no-hand cooldown
- **Configurable behavior** through environment variables

## Requirements

- Python 3.9+
- A [RoBoFlow API key](https://app.roboflow.com/)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`).

## Running

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Mobile Support

The app works on mobile browsers. Notable behavior on iPhone / Safari (portrait):

- The video feed (`#output`) fills the entire screen using `object-fit: cover`
  and `height: 100dvh`, so it stays centered and edge-to-edge even while the
  Safari address bar shows and hides (no more low-ratio, bottom-stuck
  "thumbnail").
- The color palette (`#colorBar`) is a full-width, horizontally scrollable bar
  (native momentum scrolling with a hidden scrollbar). All 11 color swatches are
  reachable on narrow portrait viewports by scrolling sideways.
- The camera uses `playsinline` (`webkit-playsinline`), so the live feed stays
  in the page instead of launching the fullscreen native player. Tap
  **Try It On** and grant camera permission when prompted.
- On mobile/tablet devices only, a flip-camera button (top-right corner) lets
  you toggle between the rear and front cameras. The active WebSocket session
  is reused, so your color/opacity settings are preserved across switches.

The frontend is a single static file at `static/index.html`, served by the
FastAPI app at `/`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROBOFLOW_API_KEY` | — | **Required.** RoBoFlow API key |
| `NAIL_ALPHA` | `0.8` | Blend strength for color transfer |
| `NAIL_BLUR` | `2` | Gaussian blur radius applied to the nail mask |
| `SPACE_DETECTION_THRESHOLD` | `0.1` | Fingertip-to-nail distance threshold |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for nail predictions |
| `MAX_DETECTION_DIM` | `320` | Max pixel dimension for MediaPipe hand detection. Lower = faster. Set to `0` to disable. |
| `MAX_PROCESS_FPS` | `20` | Max frames per second the server will process per WebSocket connection. |
| `NO_HAND_COOLDOWN` | `1.0` | Seconds to skip processing after no hands are detected. |
| `ROBOFLOW_MAX_DIM` | `640` | Max pixel dimension for images sent to the RoBoFlow API. Lower = faster API response. |

## Performance

The app is optimized for CPU-limited hosting (e.g., Render.com free tier):

- **Frame rate limiting:** Each WebSocket connection is capped at `MAX_PROCESS_FPS` to avoid CPU saturation.
- **Downscaled hand detection:** MediaPipe runs on a smaller image (`MAX_DETECTION_DIM`), then maps normalized coordinates back to the original frame.
- **No-hand cooldown:** When no hands are detected, processing is skipped for `NO_HAND_COOLDOWN` seconds.
- **Downscaled API requests:** Images sent to RoBoflow are resized to `ROBOFLOW_MAX_DIM`, then polygon coordinates are scaled back.
- **Single image decode:** Each frame is decoded from JPEG bytes only once and reused across hand detection, API inference, and painting.

## Docker

```bash
docker build -t nail-try-on .
docker run -p 8000:8000 \
  -e ROBOFLOW_API_KEY=<your-key> \
  -e NAIL_ALPHA=0.8 \
  -e NAIL_BLUR=2 \
  -e SPACE_DETECTION_THRESHOLD=0.5 \
  -e YOLO_CONFIDENCE_THRESHOLD=0.5 \
  -e MAX_DETECTION_DIM=320 \
  -e MAX_PROCESS_FPS=20 \
  -e NO_HAND_COOLDOWN=1.0 \
  -e ROBOFLOW_MAX_DIM=640 \
  nail-try-on
```

## Render

A `render.yaml` is included. Connect the repo on [Render.com](https://render.com), set `ROBOFLOW_API_KEY` in the dashboard, and deploy.

## License

MIT
