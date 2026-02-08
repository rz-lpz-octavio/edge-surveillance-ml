import os
from dotenv import load_dotenv

load_dotenv()

# Stream RTSP
RTSP_URL = os.getenv("RTSP_URL")

# Modelo
MODEL_NAME = "yolov8n.pt"

# Detección
CONFIDENCE_THRESHOLD = 0.5
