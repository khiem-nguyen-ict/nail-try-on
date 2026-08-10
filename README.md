# Nail Try-On

A real-time nail polish try-on web app built with FastAPI, MediaPipe, and RoBoFlow. The frontend captures video frames via WebSocket, detects hands and nail regions, applies color transfer to the detected nails, and streams the result back — all without leaving the browser UI.

## Features

- **Real-time processing** via WebSocket (`/ws/{session_id}`)
- **Hand detection** using MediaPipe Hands
- **Nail segmentation** using a RoBoFlow RF-DETR segmentation model
- **Color transfer** to recolor detected nail regions while preserving texture
- **Performance optimizations** for shared hosting: frame rate limiting, downscaled detection, and no-hand cooldown
- **Configurable behavior** through environment variables

## Best Output Samples
(Please refer to the nails_beauty.ipynb notebook.)
### Input Image
![Input Image](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand.webp?raw=true)

### Generated Samples (Slow Method)
![Sample 1](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand-nails-1.webp?raw=true)
![Sample 2](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand-nails-2.webp?raw=true)

## Requirements

- Python 3.11+
- A [RoBoFlow API key](https://app.roboflow.com/) - because I don't have the strong GPU hosting

## Setup

```bash
python3.12 -m venv .venv
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
| `NAIL_GLOSS_INTENSITY` | `0.5` | Gloss intensity for the painted nails (`0.0`: off, `1.0`: maximum gloss) |
| `SPACE_DETECTION_THRESHOLD` | `0.1` | Fingertip-to-nail distance threshold |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for nail predictions |
| `MAX_DETECTION_DIM` | `320` | Max pixel dimension for MediaPipe hand detection. Lower = faster. Set to `0` to disable. |
| `MAX_PROCESS_FPS` | `20` | Max frames per second the server will process per WebSocket connection. |
| `NO_HAND_COOLDOWN` | `1.0` | Seconds to skip processing after no hands are detected. |
| `ROBOFLOW_MAX_DIM` | `1024` | Max pixel dimension for images sent to the RoBoFlow API. Lower = faster API response. |
| `FRAME_SKIPPED_BLUR_THRESHOLD` | `50.0` | Laplacian variance threshold below which a frame is considered too blurry and skipped. Lower = stricter (more frames skipped). |
| `MAX_CAPTURE_DIM` | `1280` | Max pixel dimension for the captured frame sent from the browser to the backend. |
| `MAX_SEND_FPS` | `10` | Max frames per second the browser will send to the backend over WebSocket. |
| `IMAGE_QUALITY` | `80` | JPEG compression quality (0–100) for frames sent from the backend to the browser. Lower = smaller payload, lower latency. |

## Processing Workflow

Each frame from the browser WebSocket goes through this pipeline:

1. **Receive frame** — The browser sends a JPEG frame over `/ws/{session_id}`.
2. **Blur check** — The frame is evaluated using Laplacian variance. If it is below
   `FRAME_SKIPPED_BLUR_THRESHOLD`, the original frame is returned immediately and
   processing is skipped, saving CPU.
3. **Decode once** — The frame is decoded from JPEG bytes into a PIL Image.
4. **Hand detection** — MediaPipe Hands runs on a downscaled version of the frame
   (`MAX_DETECTION_DIM`) to detect hands and extract fingertip landmarks.
   If no hands are found, the original frame is returned.
5. **Nail segmentation** — The full frame is sent to the RoBoFlow RF-DETR
   segmentation model, which returns nail region polygons.
6. **Filter by proximity** — Only nail predictions that contain at least one
   fingertip are kept, using a configurable distance threshold
   (`SPACE_DETECTION_THRESHOLD`).
 7. **Paint nails** — The selected nail regions are recolored using OpenCV color
    transfer (HSV hue replacement) with configurable opacity (`NAIL_ALPHA`)
    and blur (`NAIL_BLUR`). A distance-transform-based glossy effect is applied
    using `NAIL_GLOSS_INTENSITY` to simulate light reflection on the nail surface.
 8. **Stream back** — The processed JPEG is sent back to the browser over the
    same WebSocket.

If processing fails or no nails are found, the original frame is returned.
At the WebSocket level, when no hands are detected, processing is skipped for
`NO_HAND_COOLDOWN` seconds to save CPU.

## Performance

> **Note:** This project is a proof-of-concept (PoC) demo. Frame rate and
> responsiveness depend heavily on the host's CPU. Running on better hosting
> with more CPU (e.g., Render Standard/Pro, a VPS, or local desktop) will
> deliver significantly smoother real-time performance.

The app includes several optimizations for CPU-limited hosting:

- **Frontend frame cap:** The browser is capped at `MAX_SEND_FPS` (10 FPS) so the client does not flood the WebSocket faster than the server can process.
- **Capture downscale:** Frames sent from the browser are downscaled to `MAX_CAPTURE_DIM` (1280px) to reduce bandwidth and backend processing time.
- **Frame rate limiting:** Each WebSocket connection is capped at `MAX_PROCESS_FPS` to avoid CPU saturation.
- **Blur skip:** Frames with low Laplacian variance are detected as blurry and returned unprocessed, skipping the expensive hand detection and API calls.
- **Downscaled hand detection:** MediaPipe runs on a smaller image (`MAX_DETECTION_DIM`), then maps normalized coordinates back to the original frame.
- **No-hand cooldown:** When no hands are detected, processing is skipped for `NO_HAND_COOLDOWN` seconds.
- **Downscaled API requests:** Images sent to RoBoflow are resized to `ROBOFLOW_MAX_DIM`, then polygon coordinates are scaled back.
- **Single image decode:** Each frame is decoded from JPEG bytes only once and reused across hand detection, API inference, and painting.

### Render free tier expectations

When deployed on Render's **Free** plan (512 MB RAM / 0.1 CPU), this app is
typically limited to roughly **2–3 FPS** after optimizations. The main reasons:

- **0.1 CPU share:** MediaPipe hand detection and the per-frame OpenCV painting
  pipeline are CPU-bound. Render's free tier provides only ~10% of a single
  vCPU, which throttles every processed frame.
- **Memory headroom:** 512 MB leaves little room for model loading + frame
  buffering + PIL/NumPy overhead.
- **15-minute spin-down:** Free web services spin down after 15 minutes of
  inactivity. The next request triggers a ~30–60s cold start, after which
  performance returns to the limited steady-state rate.

For production use, upgrade to at least the **Starter** plan ($7/mo, 0.5 CPU)
or **Standard** plan ($25/mo, 1 CPU). A local desktop with 4+ cores typically
runs this workload at real-time rates (15–30 FPS).

## Docker

The Docker build installs dependencies from pre-built binary wheels only
(`--only-binary=:all:`) to avoid partial or broken source builds of
`mediapipe` on Linux.

```bash
docker build -t nail-try-on .
docker run -p 8000:8000 \
  -e ROBOFLOW_API_KEY=<your-key> \
  -e NAIL_ALPHA=0.8 \
  -e NAIL_BLUR=2 \
  -e NAIL_GLOSS_INTENSITY=0.5 \
  -e SPACE_DETECTION_THRESHOLD=0.1 \
  -e YOLO_CONFIDENCE_THRESHOLD=0.5 \
  -e MAX_DETECTION_DIM=320 \
  -e MAX_PROCESS_FPS=20 \
  -e NO_HAND_COOLDOWN=1.0 \
  -e ROBOFLOW_MAX_DIM=1024 \
  -e FRAME_SKIPPED_BLUR_THRESHOLD=50.0 \
   -e MAX_CAPTURE_DIM=1280 \
  -e MAX_SEND_FPS=10 \
  -e IMAGE_QUALITY=80 \
  nail-try-on
```

## Render

A `render.yaml` is included. Connect the repo on [Render.com](https://render.com), set `ROBOFLOW_API_KEY` in the dashboard, and deploy.

> **Note:** `mediapipe` is pinned to `0.10.13` in `requirements.txt` and the
> Dockerfile installs only binary wheels. This prevents the
> `AttributeError: module 'mediapipe' has no attribute 'solutions'` error
> that can occur with unpinned versions or partial source installs on Render.

## Troubleshooting

- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
  Ensure you are using `mediapipe==0.10.13` and that the install used binary
  wheels. Reinstall with:
  ```bash
  pip install --force-reinstall --only-binary=:all: mediapipe==0.10.13
  ```
- **MediaPipe fails to import on Linux Docker**
  The project requires system libraries `libgl1`, `libglib2.0-0`, and
  `libgomp1`. The provided `Dockerfile` installs these automatically.

## License

MIT
