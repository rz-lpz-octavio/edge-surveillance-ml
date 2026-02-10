import time
import cv2
from config import RTSP_URL, MODEL_NAME, CONFIDENCE_THRESHOLD, STREAM_WIDTH, STREAM_HEIGHT
from core.stream import StreamCapture
from core.detector import PersonDetector
from output.display import Display


def main():
    stream = StreamCapture(RTSP_URL, STREAM_WIDTH, STREAM_HEIGHT)
    detector = PersonDetector(MODEL_NAME, CONFIDENCE_THRESHOLD)
    display = Display()

    print("Iniciando... (presiona 'q' para salir)")

    prev_time = time.time()

    while True:
        frame = stream.read()
        if frame is None:
            print("Error: Frame incompleto")
            break

        detections = detector.detect(frame)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        if display.render(frame, detections, fps):
            break

    stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
