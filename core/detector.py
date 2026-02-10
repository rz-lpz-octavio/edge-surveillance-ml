from ultralytics import YOLO


class PersonDetector:
    """Detector de personas usando YOLOv8."""

    def __init__(self, model_name, confidence_threshold):
        """
        Inicializa el detector.

        Args:
            model_name: Nombre o ruta del modelo YOLO.
            confidence_threshold: Umbral mínimo de confianza.
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detecta personas en un frame.

        Args:
            frame: Imagen BGR como numpy array.

        Returns:
            Lista de detecciones en formato ([x, y, w, h], confianza, clase)
            compatible con DeepSORT.
        """
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[0],
            verbose=False
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            # Convertir de xyxy a xywh (esquina superior izquierda + ancho/alto)
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, 'person'))

        return detections
