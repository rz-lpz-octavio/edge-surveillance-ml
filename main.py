import subprocess
import numpy as np
import cv2
import time
from ultralytics import YOLO
from config import RTSP_URL, MODEL_NAME, CONFIDENCE_THRESHOLD

def main():
    model = YOLO(MODEL_NAME)

    command = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', RTSP_URL,
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-an',
        '-loglevel', 'quiet',
        '-'
    ]

    width, height = 960, 1088
    frame_size = width * height * 3

    process = subprocess.Popen(command, stdout=subprocess.PIPE)

    print(f"Stream: {width}x{height}")
    print("Descartando frames iniciales...")

    # Descartar primeros frames corruptos
    for _ in range(30):
        process.stdout.read(frame_size)

    print("Iniciando... (presiona 'q' para salir)")

    prev_time = time.time()

    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            print("Error: Frame incompleto")
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

        # Detección (solo personas, clase 0)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
        annotated = results[0].plot()

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Surveillance", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    process.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
