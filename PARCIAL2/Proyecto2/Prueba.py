import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la primera imagen en escala de grises
image1_path = "t1.jpg"
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)

# Aplicar filtro Gaussiano para reducción de ruido
image1_blur = cv2.GaussianBlur(image1, (5, 5), 0)

# Aplicar corrección de contraste adaptativa CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5,5))
image1_clahe = clahe.apply(image1_blur)

# Aplicar umbralización manual con un umbral específico
threshold_value1 = 140
_, thresh1 = cv2.threshold(image1_clahe, threshold_value1, 255, cv2.THRESH_BINARY)

# Aplicar operaciones morfológicas
kernel = np.ones((8, 8), np.uint8)
morph1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=2)

# Encontrar contornos
contours1, _ = cv2.findContours(morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Dibujar contornos
image1_contours = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image1_contours, contours1, -1, (255, 0, 0), 2)

# Cargar la segunda imagen en escala de grises
image2_path = "t2.jpg"
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Aplicar filtro Gaussiano para reducción de ruido
image2_blur = cv2.GaussianBlur(image2, (5, 5), 0)

# Aplicar corrección de contraste adaptativa CLAHE
image2_clahe = clahe.apply(image2_blur)

# Aplicar umbralización manual con un umbral específico
threshold_value2 = 240
_, thresh2 = cv2.threshold(image2_clahe, threshold_value2, 255, cv2.THRESH_BINARY)

# Aplicar operaciones morfológicas
morph2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel, iterations=2)

# Encontrar contornos
contours2, _ = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Dibujar contornos
image2_contours = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image2_contours, contours2, -1, (255, 0, 0), 2)

# Mostrar resultados
fig, axs = plt.subplots(2, 3, figsize=(20, 10))

axs[0, 0].imshow(image1, cmap='gray')
axs[0, 0].set_title("Imagen 1 Original")
axs[0, 0].axis("off")
axs[0, 1].imshow(morph1, cmap='gray')
axs[0, 1].set_title("Segmentación del Tumor 1")
axs[0, 1].axis("off")
axs[0, 2].imshow(image1_contours)
axs[0, 2].set_title("Detección de Tumor 1")
axs[0, 2].axis("off")

axs[1, 0].imshow(image2, cmap='gray')
axs[1, 0].set_title("Imagen 2 Original")
axs[1, 0].axis("off")
axs[1, 1].imshow(morph2, cmap='gray')
axs[1, 1].set_title("Segmentación del Tumor 2")
axs[1, 1].axis("off")
axs[1, 2].imshow(image2_contours)
axs[1, 2].set_title("Detección de Tumor 2")
axs[1, 2].axis("off")

plt.show()
