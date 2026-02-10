import os
from dotenv import load_dotenv

load_dotenv()

# Stream RTSP
RTSP_URL = os.getenv("RTSP_URL")
STREAM_WIDTH = 960
STREAM_HEIGHT = 1088

# Modelo
MODEL_NAME = "yolov8n.pt"

# Detección
CONFIDENCE_THRESHOLD = 0.5
