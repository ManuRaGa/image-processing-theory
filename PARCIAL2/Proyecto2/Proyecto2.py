import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_imagen(image_path, threshold_value):
    # Cargar la imagen en escala de grises
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar filtro Gaussiano para reducción de ruido
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Aplicar corrección de contraste adaptativa CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image_clahe = clahe.apply(image_blur)
    
    # Aplicar umbralización manual con un umbral más selectivo
    _, thresh = cv2.threshold(image_clahe, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Aplicar operaciones morfológicas para refinar la segmentación
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos basado en área para eliminar regiones grandes no deseadas
    filtered_contours = [cnt for cnt in contours if 500 < cv2.contourArea(cnt) < 5000]
    
    # Dibujar los contornos detectados en la imagen original a color
    image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_contours, filtered_contours, -1, (0, 255, 0), 2)
    
    return image, image_clahe, morph, image_contours

# Definir umbrales específicos para cada imagen
image_paths = ["t1.jpg", "t2.jpg"]  # Cambia a las imágenes que desees procesar
threshold_values = [140, 240]  # Ajusta los valores de umbralización para cada imagen

# Procesar ambas imágenes con umbrales independientes
results = [procesar_imagen(img, thr) for img, thr in zip(image_paths, threshold_values)]

# Mostrar resultados
fig, axs = plt.subplots(len(results), 4, figsize=(20, 10))

for i, (image, image_clahe, morph, image_contours) in enumerate(results):
    axs[i, 0].imshow(image, cmap='gray')
    axs[i, 0].set_title(f"Imagen {i+1} Original")
    axs[i, 0].axis("off")
    
    axs[i, 1].imshow(image_clahe, cmap='gray')
    axs[i, 1].set_title(f"Corrección de Contraste CLAHE {i+1}")
    axs[i, 1].axis("off")
    
    axs[i, 2].imshow(morph, cmap='gray')
    axs[i, 2].set_title(f"Segmentación con Umbralización {i+1}")
    axs[i, 2].axis("off")
    
    axs[i, 3].imshow(image_contours)
    axs[i, 3].set_title(f"Detección de Tumor {i+1}")
    axs[i, 3].axis("off")

plt.tight_layout()
plt.show()
