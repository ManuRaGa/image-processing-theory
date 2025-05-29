import cv2

# Cargar la imagen original
imagen = cv2.imread("Foto.jpg")

# Redimensionar
imagen = cv2.resize(imagen, (600, 800))

# Coordenadas
coordenadas_piezas = [
    (150, 250, 150, 150),
    (300, 50, 150, 150),
    (50, 550, 150, 150),
    (450, 300, 150, 150),
]

# Crear una copia para no modificar la original
tablero = imagen.copy()

# Dibujar los huecos en blanco
for (x, y, w, h) in coordenadas_piezas:
    cv2.rectangle(tablero, (x, y), (x+w, y+h), (255, 255, 255), -1)

# Guardar la imagen del tablero
cv2.imwrite("tablero_con_huecos.png", tablero)