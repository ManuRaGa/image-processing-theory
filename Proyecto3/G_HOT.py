import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar el tablero con huecos
tablero = cv2.imread("Foto.jpg")
tablero = cv2.resize(tablero, (600, 800)) 
tablero_gray = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)

# Inicializar ORB y matcher
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

n_piezas = 4

for i in range(1, n_piezas + 1):
    pieza = cv2.imread(f"pieza_{i}.png")
    pieza = cv2.resize(pieza, (150, 150))
    pieza_gray = cv2.cvtColor(pieza, cv2.COLOR_BGR2GRAY)

    # Detectar keypoints y descriptores
    kp1, des1 = orb.detectAndCompute(pieza_gray, None)
    kp2, des2 = orb.detectAndCompute(tablero_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print(f"Pieza {i} no tiene suficientes puntos clave.")
        continue

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print(f"No hay suficientes matches para la pieza {i}.")
        continue

    # Visualizar los matches
    img_matches = cv2.drawMatches(pieza, kp1, tablero, kp2, matches[:30], None, flags=2)
    plt.figure(figsize=(25, 10))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"Matches de la Pieza {i}")
    plt.axis("off")
    plt.savefig(f"matches_pieza_{i}.png")
    plt.close()

    # Homografía con los mejores matches
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print(f"No se pudo calcular homografía para pieza {i}.")
        continue

    h_p, w_p = pieza.shape[:2]

    # Transformar la pieza
    pieza_warp = cv2.warpPerspective(pieza, H, (tablero.shape[1], tablero.shape[0]))
    mascara = cv2.warpPerspective(np.ones((h_p, w_p), dtype=np.uint8) * 255, H, (tablero.shape[1], tablero.shape[0]))

    # Colocar la pieza en el tablero
    tablero[mascara > 0] = pieza_warp[mascara > 0]

    # Dibujar el contorno de la pieza colocada
    esquinas = np.float32([[0, 0], [w_p, 0], [w_p, h_p], [0, h_p]]).reshape(-1, 1, 2)
    esquinas_transformadas = cv2.perspectiveTransform(esquinas, H).astype(int)
    color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
    cv2.polylines(tablero, [esquinas_transformadas], isClosed=True, color=color, thickness=3)
    cv2.putText(tablero, f"Pieza {i}",
                tuple(esquinas_transformadas[0][0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Mostrar y guardar el paso
    cv2.imshow(f"Paso {i}", tablero)
    cv2.imwrite(f"paso_{i}_orb.png", tablero)
    print(f"Pieza {i} colocada. Guardada como paso_{i}_orb.png")
    cv2.waitKey(0)
    cv2.destroyWindow(f"Paso {i}")

# Guardar resultado final
cv2.imwrite("rompecabezas_final_orb.png", tablero)
cv2.imshow("Rompecabezas Final (ORB)", tablero)
cv2.waitKey(0)
cv2.destroyAllWindows()