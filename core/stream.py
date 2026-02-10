import subprocess
import numpy as np


class StreamCapture:
    """Captura de video RTSP usando ffmpeg como subproceso."""

    def __init__(self, rtsp_url, width, height, discard_frames=30):
        """
        Inicializa la captura de stream RTSP.

        Args:
            rtsp_url: URL del stream RTSP.
            width: Ancho del frame en píxeles.
            height: Alto del frame en píxeles.
            discard_frames: Frames iniciales a descartar (corruptos).
        """
        self.width = width
        self.height = height
        self.frame_size = width * height * 3

        command = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
            '-loglevel', 'quiet',
            '-'
        ]

        self.process = subprocess.Popen(command, stdout=subprocess.PIPE)

        print(f"Stream: {width}x{height}")
        print("Descartando frames iniciales...")

        # Descartar primeros frames corruptos
        for _ in range(discard_frames):
            self.process.stdout.read(self.frame_size)

    def read(self):
        """
        Lee un frame del stream.

        Returns:
            numpy.ndarray con el frame, o None si falla la lectura.
        """
        raw = self.process.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

    def release(self):
        """Termina el proceso de ffmpeg."""
        self.process.terminate()
