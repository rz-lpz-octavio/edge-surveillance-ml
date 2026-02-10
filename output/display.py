import cv2


class Display:
    """Renderizado de video con bounding boxes, IDs y alertas."""

    WINDOW_NAME = "Surveillance"

    def render(self, frame, tracks, alerts, fps):
        """
        Dibuja la información de tracking y alertas sobre el frame.

        Args:
            frame: Frame BGR original.
            tracks: Lista de dicts con track_id, bbox, centroid.
            alerts: Lista de alertas activas.
            fps: FPS actual del pipeline.

        Returns:
            True si el usuario presionó 'q' (salir), False en caso contrario.
        """
        display_frame = frame.copy()
        alert_ids = {a['track_id'] for a in alerts}

        for track in tracks:
            tid = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            is_loitering = tid in alert_ids

            # Color: rojo si merodea, verde si normal
            color = (0, 0, 255) if is_loitering else (0, 255, 0)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con ID
            label = f"ID:{tid}"
            if is_loitering:
                label += " MERODEO"

            cv2.putText(
                display_frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # FPS
        cv2.putText(
            display_frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        cv2.imshow(self.WINDOW_NAME, display_frame)

        return (cv2.waitKey(1) & 0xFF) == ord('q')
