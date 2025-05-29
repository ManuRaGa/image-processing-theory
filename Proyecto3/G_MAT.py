import cv2
import numpy as np

# Cargar el tablero con huecos
tablero = cv2.imread("tablero_con_huecos.png")

coordenadas_piezas = [
    (150, 250, 150, 150),
    (300, 50, 150, 150),
    (50, 550, 150, 150),
    (450, 300, 150, 150),
]

for i, (x, y, w, h) in enumerate(coordenadas_piezas):
    pieza = cv2.imread(f"pieza_{i+1}.png")

    # Colocar la pieza directamente
    tablero[y:y+h, x:x+w] = pieza

    # Dibujar contorno y etiqueta
    color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    cv2.rectangle(tablero, (x, y), (x+w, y+h), color, 3)
    cv2.putText(tablero, f"Pieza {i+1}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar y guardar paso
    cv2.imshow(f"Paso {i+1}", tablero)
    cv2.imwrite(f"paso_{i+1}.png", tablero)
    print(f"Pieza {i+1} colocada. Imagen guardada como paso_{i+1}.png")
    cv2.waitKey(0)
    cv2.destroyWindow(f"Paso {i+1}")

# Guardar resultado final
cv2.imwrite("rompecabezas_final_directo.png", tablero)
cv2.imshow("Rompecabezas Final", tablero)
cv2.waitKey(0)
cv2.destroyAllWindows()