import time
from utils.geometry import convex_hull_area


class BehaviorAnalyzer:
    """Analiza comportamiento de personas rastreadas (merodeo)."""

    def __init__(self, loiter_time, loiter_area, max_missing_frames=90,
                 window_size=50):
        """
        Inicializa el analizador de comportamiento.

        Args:
            loiter_time: Tiempo mínimo (segundos) para considerar merodeo.
            loiter_area: Área máxima del convex hull (px²) para merodeo.
            max_missing_frames: Frames sin ver un track antes de eliminarlo.
            window_size: Cantidad de centroides recientes a considerar.
        """
        self.loiter_time = loiter_time
        self.loiter_area = loiter_area
        self.max_missing_frames = max_missing_frames
        self.window_size = window_size
        # track_id → {centroids, first_seen, last_seen_frame}
        self.tracks = {}
        self.frame_count = 0

    def update(self, tracked_objects):
        """
        Actualiza el análisis con los tracks actuales.

        Args:
            tracked_objects: Lista de dicts con track_id, bbox, centroid.

        Returns:
            Lista de alertas: {track_id, type, elapsed_time, area}
        """
        self.frame_count += 1
        now = time.time()

        alerts = []

        for obj in tracked_objects:
            tid = obj['track_id']
            centroid = obj['centroid']

            if tid not in self.tracks:
                self.tracks[tid] = {
                    'centroids': [],
                    'first_seen': now,
                    'last_seen_frame': self.frame_count
                }

            self.tracks[tid]['centroids'].append(centroid)
            # Ventana deslizante: solo los últimos N centroides
            if len(self.tracks[tid]['centroids']) > self.window_size:
                self.tracks[tid]['centroids'] = self.tracks[tid]['centroids'][-self.window_size:]
            self.tracks[tid]['last_seen_frame'] = self.frame_count

            # Evaluar condición de merodeo
            elapsed = now - self.tracks[tid]['first_seen']
            if elapsed >= self.loiter_time:
                area = convex_hull_area(self.tracks[tid]['centroids'])
                if area < self.loiter_area:
                    alerts.append({
                        'track_id': tid,
                        'type': 'loitering',
                        'elapsed_time': elapsed,
                        'area': area
                    })

        # Limpiar tracks que ya no se ven
        stale = [
            tid for tid, data in self.tracks.items()
            if self.frame_count - data['last_seen_frame'] > self.max_missing_frames
        ]
        for tid in stale:
            del self.tracks[tid]

        return alerts
