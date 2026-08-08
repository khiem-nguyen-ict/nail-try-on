# Nail Try-On

A real-time nail polish try-on web app built with FastAPI, MediaPipe, and RoBoFlow. The frontend captures video frames via WebSocket, detects hands and nail regions, applies color transfer to the detected nails, and streams the result back — all without leaving the browser UI.

## Features

- **Real-time processing** via WebSocket (`/ws/{session_id}`)
- **Hand detection** using MediaPipe Hands
- **Nail segmentation** using a RoBoFlow RF-DETR segmentation model
- **Color transfer** to recolor detected nail regions while preserving texture
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

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROBOFLOW_API_KEY` | — | **Required.** RoBoFlow API key |
| `NAIL_ALPHA` | `0.8` | Blend strength for color transfer |
| `NAIL_BLUR` | `2` | Gaussian blur radius applied to the nail mask |
| `SPACE_DETECTION_THRESHOLD` | `0.5` | Fingertip-to-nail distance threshold |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for nail predictions |

## Docker

```bash
docker build -t nail-try-on .
docker run -p 8000:8000 \
  -e ROBOFLOW_API_KEY=<your-key> \
  -e NAIL_ALPHA=0.8 \
  -e NAIL_BLUR=2 \
  -e SPACE_DETECTION_THRESHOLD=0.5 \
  -e YOLO_CONFIDENCE_THRESHOLD=0.5 \
  nail-try-on
```

## Render

A `render.yaml` is included. Connect the repo on [Render.com](https://render.com), set `ROBOFLOW_API_KEY` in the dashboard, and deploy.

## License

MIT
