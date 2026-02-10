import cv2


class Display:
    """Renderizado de video con bounding boxes y FPS."""

    WINDOW_NAME = "Surveillance"

    def render(self, frame, detections, fps):
        """
        Dibuja detecciones y FPS sobre el frame.

        Args:
            frame: Frame BGR original.
            detections: Lista de ([x, y, w, h], confianza, clase).
            fps: FPS actual del pipeline.

        Returns:
            True si el usuario presionó 'q' (salir), False en caso contrario.
        """
        display_frame = frame.copy()

        for (bbox, conf, _cls) in detections:
            x1, y1, w, h = [int(v) for v in bbox]
            cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(
                display_frame, f"{conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # FPS
        cv2.putText(
            display_frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow(self.WINDOW_NAME, display_frame)

        return (cv2.waitKey(1) & 0xFF) == ord('q')
