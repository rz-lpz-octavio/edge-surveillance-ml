import subprocess
import numpy as np
import cv2
import time
from ultralytics import YOLO
from config import RTSP_URL, MODEL_NAME, CONFIDENCE_THRESHOLD

def main():
    model = YOLO(MODEL_NAME)

    # Usar ffmpeg con TCP para capturar frames
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

    prev_time = time.time()

    print(f"Stream: {width}x{height}")
    print("Descartando frames iniciales...")

    # Descartar primeros 30 frames (corruptos)
    for i in range(30):
        process.stdout.read(frame_size)

    print("Iniciando captura... (Ctrl+C para salir)")
    print("Guardando video en output.mp4...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10, (width, height))

    frame_count = 0
    max_frames = 100

    try:
        while frame_count < max_frames:
            raw_frame = process.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                print("Error: Frame incompleto")
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3))

            # Guardar primer frame para debug
            if frame_count == 0:
                cv2.imwrite('debug_frame.jpg', frame)
                print(f"Debug: min={frame.min()}, max={frame.max()}")

            # Detección (solo personas, clase 0)
            results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)

            # Dibujar detecciones
            annotated = results[0].plot()

            # Calcular FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(annotated)
            frame_count += 1

            if frame_count % 20 == 0:
                print(f"Frames: {frame_count}/{max_frames}, FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrumpido")

    out.release()
    process.terminate()
    print(f"Video: output.mp4 ({frame_count} frames)")

if __name__ == "__main__":
    main()
