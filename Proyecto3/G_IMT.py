import cv2

# Cargar la imagen base
imagen = cv2.imread("Foto.jpg")

# Redimensionar por si está muy grande
imagen = cv2.resize(imagen, (600, 800))

# Crear lista de coordenadas para 4 piezas (x, y, ancho, alto)
coordenadas_piezas = [
    (150, 250, 150, 150),
    (300, 50, 150, 150),
    (50, 550, 150, 150),
    (450, 300, 150, 150),
]

for i, (x, y, w, h) in enumerate(coordenadas_piezas):
    # Recortar la pieza
    pieza = imagen[y:y+h, x:x+w]

    # Aplicar una transformación diferente a cada pieza
    if i == 0:
        pieza = cv2.rotate(pieza, cv2.ROTATE_90_CLOCKWISE)
    elif i == 1:
        pieza = cv2.GaussianBlur(pieza, (5, 5), 0)
    elif i == 2:
        pieza = cv2.resize(pieza, (int(w*0.8), int(h*0.8)))
    elif i == 3:
        matriz_rot = cv2.getRotationMatrix2D((w//2, h//2), 45, 1)
        pieza = cv2.warpAffine(pieza, matriz_rot, (w, h))

    # Guardar la pieza
    cv2.imwrite(f"pieza_{i+1}.png", pieza)