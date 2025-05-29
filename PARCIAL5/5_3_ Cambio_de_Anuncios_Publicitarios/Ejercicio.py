import cv2
import numpy as np

puntos = []

img_fondo = cv2.imread("Fondo.jpg")
anuncio = cv2.imread("F1.jpg")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
        puntos.append((x, y))
        cv2.circle(img_fondo, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Puntos", img_fondo)

        if len(puntos) == 4:
            cv2.destroyAllWindows()
            insertar_anuncio()

def insertar_anuncio():
    h, w, _ = anuncio.shape
    pts_anuncio = np.float32([[0,0], [w,0], [w,h], [0,h]])
    
    pts_destino = np.float32(puntos)

    matrix = cv2.getPerspectiveTransform(pts_anuncio, pts_destino)
    anuncio_warped = cv2.warpPerspective(anuncio, matrix, (img_fondo.shape[1], img_fondo.shape[0]))

    mask = np.zeros_like(img_fondo, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_destino), (255, 255, 255))
    fondo_masked = cv2.bitwise_and(img_fondo, cv2.bitwise_not(mask))
    resultado = cv2.add(fondo_masked, anuncio_warped)

    cv2.imshow("Resultado", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imshow("Puntos", img_fondo)
cv2.setMouseCallback("Puntos", click_event)
cv2.waitKey(0)
