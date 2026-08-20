# Nail Try-On

A real-time nail polish try-on web app built with FastAPI, MediaPipe, and RoBoFlow. The frontend captures video frames via WebSocket, detects hands and nail regions, applies color transfer to the detected nails, and streams the result back — all without leaving the browser UI.

## Features

- **Real-time processing** via WebSocket (`/ws/{session_id}`)
- **Hand detection** using MediaPipe Hands
- **Nail segmentation** using a RoBoFlow RF-DETR segmentation model
- **Color transfer** to recolor detected nail regions with configurable opacity and a distance-transform glossy effect
- **Performance optimizations** for shared hosting: frame rate limiting, downscaled detection, and no-hand cooldown
- **Configurable behavior** through environment variables

## Models

This project relies on two machine learning components:

### MediaPipe Hands

Hand detection is performed using [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker). The model runs locally in the backend and returns 21 hand landmarks per detected hand, from which the five fingertip positions are extracted.

- **Library version:** `mediapipe==0.10.13`
- **Configuration:** `static_image_mode=True`, `max_num_hands=2`, `model_complexity=1`, `min_detection_confidence=0.85`, `min_tracking_confidence=0.85`

### RoBoFlow RF-DETR Nail Segmentation

Nail regions are segmented using a [RoBoFlow](https://roboflow.com/) serverless inference endpoint running the **RF-DETR** (Real-Time Detection Transformer) segmentation model. The model is hosted on RoBoFlow's infrastructure and accessed via HTTP POST.

- **Model:** `thanh-khiem-nguyen/nails_segmentation-m8ew1-1-rfdetr-seg-large-t1`
- **Output:** Polygon masks for detected nail regions with confidence scores
- **Filtering:** Every detected nail is matched to its nearest fingertip via a one-to-one assignment (no hand-tuned proximity threshold). Nails without a nearby fingertip are dropped (not painted with a fallback orientation), and each matched fingertip is marked (`matched: True`) in `hands_data` so downstream consumers can inspect which tips were consumed.
- **Optimization:** Images sent to the API are downscaled to `ROBOFLOW_MAX_DIM` to reduce latency and response size
- **Debug output:** Passing `debug_save_path` to `detect_hands()` generates a visualization with fingertip (red), DIP (green), dashed connecting lines, and finger ID labels for manual verification.

## Pipeline Steps

The `experiments/static_painting.py` script demonstrates each step of the
offline nail painting pipeline on the same input image:

### 1. Original Image
Input image fed into the pipeline.

![Original](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand.webp?raw=true)

### 2. Hand Landmark Detection
MediaPipe Hands detects 21 hand landmarks per hand. Fingertip landmarks (red)
and DIP landmarks (green) are drawn with dashed white connecting lines and
finger ID labels. This step extracts finger angles, 3D angles (`a3d`), and
depth (`z`) for each fingertip.

![Hand Debug](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand-output-mp-debug.webp?raw=true)

### 3. Nail Segmentation Mask
RoBoFlow RF-DETR segments nail regions as polygons. Each polygon is matched to
its nearest fingertip using a one-to-one assignment so every fingertip is
paired with at most one nail. Nails that cannot be paired with a nearby
fingertip are dropped rather than painted with a fallback orientation.

![Nail Mask](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand-output-mask.webp?raw=true)

### 4. Final Painted Result
The selected nail regions are recolored with full HSV color transfer, blended
at `NAIL_ALPHA` and blurred with `NAIL_BLUR`. A distance-transform-based
glossy effect simulates light reflection on the nail surface. Nails are sorted
by depth (`z` sum) and 3D angle (`a3d`) before painting so overlapping nails
render in correct back-to-front order.

![Final Result](https://github.com/khiem-nguyen-ict/nail-try-on/blob/main/sample-images/hand-output.webp?raw=true)

## Experiments

The `experiments/` directory contains standalone scripts (not used by the live
server) that explore the nail-painting pipeline end-to-end:

### `static_painting.py`

The `experiments/static_painting.py` script demonstrates offline nail painting
using a reference pattern image and the RoBoFlow segmentation mask. It delegates
the per-nail painting to `app.services.nail_pattern_painter.paint_nail_pattern`
and supports depth-based lighting using the MediaPipe `z` coordinate:

- **Depth-based brightness:** Fingertip `z` values (relative to the wrist) are
  mapped to brightness adjustments so nails closer to the camera appear brighter
  and those further away appear dimmer.
- **Color matching:** Nail colors are subtly adjusted to harmonize with the base
  image HSV profile.
- **Nail depth sorting:** Nails are sorted by depth (`z` sum of polygon points)
  and 3D angle (`a3d`) before painting so overlapping nails render in correct
  back-to-front order.
- **MediaPipe debug output:** When `detect_hands()` is called with
  `debug_save_path`, it saves a visualization showing fingertip landmarks (red),
  DIP landmarks (green), dashed white connecting lines, and finger ID labels.
  The script writes these as `*-mp-debug.*` files alongside the painted outputs.
- **Outputs:** Generates a painted result, a debug mask, and a MediaPipe debug
  image under `sample-images/`.

### `model_painting.py`

`experiments/model_painting.py` experiments with a generative approach using
Stable Diffusion XL + ControlNet + inpainting, blended with the pattern from
`sample-images/sample.png` onto the segmented nail mask (handedness and
fingertip angles are used to orient each pattern). This is research-only and
requires `diffusers` + `torch` plus a `ROBOFLOW_API_KEY`.

### `nails_beauty.ipynb`

`experiments/nails_beauty.ipynb` is an exploratory Jupyter notebook covering
alternative nail-beauty workflows (color transfer, gloss, and pattern overlay).

## Project Structure

The backend is organized as a Python package under `app/` to separate
configuration, services, utilities, and API wiring:

```
app/
  __init__.py
  api/
    __init__.py          # API package (reserved for route modules)
  config.py              # Environment variables, constants, and RoBoFlow config
  main.py                # FastAPI app factory, routes, and ASGI entrypoint
  services/
    __init__.py
    frame_processor.py    # Frame pipeline orchestration
    hand_detector.py      # MediaPipe hand detection and blur check
    nail_detector.py      # RoBoFlow API client and nail filtering
    nail_painter.py       # HSV color-transfer + distance-transform glossy paint
    nail_pattern_painter.py  # Reference-pattern painting (perspective/skew/depth)
  utils/
    __init__.py
    color.py             # Hex-to-RGB color helper
    image.py             # Color matching and base image profile helpers
    polygon.py           # Nail geometry helpers (rotated rectangle, sizing)
```

- **`app/config.py`** loads environment variables via `python-dotenv` and
  exposes typed constants used across the app.
- **`app/api/`** is a package reserved for route modules.
- **`app/services/`** contains the processing pipeline:
  hand detection, nail segmentation, nail painting, and frame orchestration.
  `nail_painter.py` recolors nails via HSV transfer + a glossy mask, while
  `nail_pattern_painter.py` stamps a reference pattern (e.g. a nail-art
  swatch) onto nails using perspective skew, depth-based brightness, and color
  matching. `frame_processor.py` orchestrates these steps per frame.
- **`app/utils/`** holds small pure helpers shared by the services:
  `color.py` (color conversion), `image.py` (color-matching against the
  base image's HSV profile), and `polygon.py` (nail-geometry helpers such as
  rotated bounding-rectangle computation and nail sizing).
- **`app/main.py`** defines the FastAPI app, mounts static files, and
  registers the HTTP and WebSocket routes. The ASGI app object is exported
  as `app.main:app`.

The root-level `app.py` was removed to avoid shadowing the `app/` package.
The Docker entrypoint and local run command now reference `app.main:app`.

## Requirements

- Python 3.11+
- A [RoBoFlow API key](https://app.roboflow.com/) - because I don't have the strong GPU hosting

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`).

## Running

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
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
| `NAIL_ALPHA` | `0.4` | Blend strength for color transfer |
| `NAIL_BLUR` | `1` | Gaussian blur radius applied to the nail mask |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for nail predictions |
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
4. **Hand detection** — MediaPipe Hands runs on the frame in static-image mode
   (`static_image_mode=True`, `model_complexity=1`) with detection and tracking
   confidence at `0.85` to extract fingertip landmarks.
   If no hands are found, the original frame is returned.
5. **Nail segmentation** — The full frame is sent to the RoBoFlow RF-DETR
   segmentation model, which returns nail region polygons.
 6. **Match nails to fingers** — Each detected nail is matched to its nearest
    fingertip via a one-to-one assignment, so every fingertip is paired with
    at most one nail (no finger is shared between nails). Nails that cannot be
    paired with a nearby fingertip are dropped. Each matched fingertip is marked
    in `hands_data` (`matched: True`).
  7. **Paint nails** — The selected nail regions are recolored using full HSV
     color transfer with the selected color, blended at `NAIL_ALPHA` and blurred
     with `NAIL_BLUR`. A distance-transform-based glossy effect is applied
     to simulate light reflection on the nail surface.
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
  -e NAIL_ALPHA=0.4 \
  -e NAIL_BLUR=1 \
  -e YOLO_CONFIDENCE_THRESHOLD=0.5 \
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
