import cv2
import numpy as np
import matplotlib.pyplot as plt

# === 1. Cargar imágenes: tablero con huecos y una pieza ===
tablero = cv2.imread("Foto.jpg")
pieza = cv2.imread("pieza_1.png")

# Redimensionar opcionalmente si las piezas fueron recortadas de una imagen escalada
tablero = cv2.resize(tablero, (600, 800)) 
pieza = cv2.resize(pieza, (150, 150))

# Escala de grises
tablero_g = cv2.cvtColor(tablero, cv2.COLOR_BGR2GRAY)
pieza_g = cv2.cvtColor(pieza, cv2.COLOR_BGR2GRAY)

# === 2. ORB keypoints y descriptores ===
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(pieza_g, None)
kp2, des2 = orb.detectAndCompute(tablero_g, None)

# Visualizar keypoints
img1_kp = cv2.drawKeypoints(pieza, kp1, None, color=(255, 255, 255))
img2_kp = cv2.drawKeypoints(tablero, kp2, None, color=(255, 255, 255))

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
plt.title("Keypoints de la pieza")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
plt.title("Keypoints del tablero")
plt.axis("off")
plt.show()

# === 3. Matching con BFMatcher ===
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Visualizar los matches
img_matches = cv2.drawMatches(pieza, kp1, tablero, kp2, matches[:30], None, flags=2)
plt.figure(figsize=(30, 20))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Mejores matches entre pieza y tablero")
plt.axis("off")
plt.show()

# === 4. Calcular homografía con los mejores matches ===
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:30]]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print("Homografía calculada:")
print(H)

# === 5. WarpPerspective para transformar la pieza ===
h_p, w_p = pieza.shape[:2]
corners = np.float32([[0, 0], [w_p, 0], [0, h_p], [w_p, h_p]]).reshape(-1, 1, 2)
transformed_corners = cv2.perspectiveTransform(corners, H)

# Crear imagen nueva del mismo tamaño que el tablero
pieza_warp = cv2.warpPerspective(pieza, H, (tablero.shape[1], tablero.shape[0]))

# Crear máscara y combinar con tablero
mascara = cv2.warpPerspective(np.ones((h_p, w_p), dtype=np.uint8)*255, H, (tablero.shape[1], tablero.shape[0]))
tablero_colocado = tablero.copy()
tablero_colocado[mascara > 0] = pieza_warp[mascara > 0]

# Dibujar contorno en la pieza colocada
cv2.polylines(tablero_colocado, [np.int32(transformed_corners)], isClosed=True, color=(0,255,0), thickness=3)
cv2.putText(tablero_colocado, "Pieza 1", tuple(np.int32(transformed_corners[0][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# Mostrar resultado final
plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(tablero_colocado, cv2.COLOR_BGR2RGB))
plt.title("Pieza colocada con homografía")
plt.axis("off")
plt.show()
