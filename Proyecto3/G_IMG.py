import cv2

# Cargar la imagen base
imagen = cv2.imread("Foto.jpg")

# Redimensionar
imagen = cv2.resize(imagen, (600, 800))

# Coordenadas (x, y, ancho, alto)
coordenadas_piezas = [
    (150, 250, 150, 150),
    (300, 50, 150, 150),
    (50, 550, 150, 150),
    (450, 300, 150, 150),
]

for i, (x, y, w, h) in enumerate(coordenadas_piezas):
    pieza = imagen[y:y+h, x:x+w]
    cv2.imwrite(f"pieza_{i+1}.png", pieza)