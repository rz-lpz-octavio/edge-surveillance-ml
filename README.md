# edge-surveillance-ml

Real-time person detection for home surveillance using YOLOv8.

## Requirements

- Python 3.12 (3.13 not supported by ultralytics)
- RTSP camera stream

## Setup

```bash
conda activate surveillance
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your RTSP credentials
```

## Usage

```bash
conda activate surveillance
python main.py
```
