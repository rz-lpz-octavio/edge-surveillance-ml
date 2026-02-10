import numpy as np


def convex_hull_area(points):
    """
    Calcula el área del convex hull de un conjunto de puntos 2D.

    Usa el algoritmo Gift Wrapping + fórmula Shoelace para evitar
    dependencias externas (scipy).

    Args:
        points: Lista de tuplas (x, y).

    Returns:
        Área del convex hull en píxeles cuadrados.
        Retorna 0.0 si hay menos de 3 puntos.
    """
    if len(points) < 3:
        return 0.0

    pts = np.array(points)

    # Eliminar puntos duplicados
    pts = np.unique(pts, axis=0)

    if len(pts) < 3:
        return 0.0

    # Gift Wrapping (Jarvis March) para obtener el convex hull
    hull = _gift_wrap(pts)

    if len(hull) < 3:
        return 0.0

    # Fórmula Shoelace para el área
    return _shoelace_area(hull)


def _gift_wrap(points):
    """
    Algoritmo Gift Wrapping para calcular el convex hull.

    Args:
        points: numpy array de puntos (N, 2).

    Returns:
        Lista de puntos que forman el convex hull en orden.
    """
    n = len(points)
    # Empezar desde el punto más a la izquierda
    start = int(np.argmin(points[:, 0]))
    hull = []
    current = start

    while True:
        hull.append(points[current])
        candidate = 0

        for i in range(n):
            if i == current:
                continue
            # Producto cruzado para determinar orientación
            cross = _cross_product(
                points[current], points[candidate], points[i]
            )
            if candidate == current or cross > 0:
                candidate = i
            elif cross == 0:
                # Si son colineales, tomar el más lejano
                d1 = np.sum((points[i] - points[current]) ** 2)
                d2 = np.sum((points[candidate] - points[current]) ** 2)
                if d1 > d2:
                    candidate = i

        current = candidate
        if current == start:
            break

    return np.array(hull)


def _cross_product(o, a, b):
    """Producto cruzado de vectores OA y OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _shoelace_area(hull):
    """
    Calcula el área de un polígono usando la fórmula Shoelace.

    Args:
        hull: numpy array de vértices del polígono en orden.

    Returns:
        Área del polígono.
    """
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0
