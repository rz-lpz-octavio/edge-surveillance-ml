from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    """Tracking persistente de objetos usando DeepSORT."""

    def __init__(self):
        """Inicializa el tracker DeepSORT."""
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        """
        Actualiza el estado del tracker con nuevas detecciones.

        Args:
            detections: Lista de ([x, y, w, h], confianza, clase).
            frame: Frame actual (usado para extracción de features).

        Returns:
            Lista de dicts con información de cada track activo:
            {track_id, bbox: [x1, y1, x2, y2], centroid: (cx, cy)}
        """
        tracks_raw = self.tracker.update_tracks(detections, frame=frame)

        tracks = []
        for track in tracks_raw:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            tracks.append({
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'centroid': (cx, cy)
            })

        return tracks
