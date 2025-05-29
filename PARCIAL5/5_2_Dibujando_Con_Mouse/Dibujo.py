import cv2
import numpy as np

dibujando = False
punto_inicial = (-1, -1)
figuras = []
tipo_figura = 'r'

color_actual = [255, 255, 255]

def mouse_evento(event, x, y, flags, param):
    global dibujando, punto_inicial, figuras

    if event == cv2.EVENT_LBUTTONDOWN:
        dibujando = True
        punto_inicial = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dibujando:
        param['preview'] = (punto_inicial, (x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        dibujando = False
        punto_final = (x, y)
        figuras.append((tipo_figura, punto_inicial, punto_final, tuple(color_actual)))
        param['preview'] = None

cv2.namedWindow("Dibujo")

def actualizar_color(x): pass

cv2.createTrackbar('R', 'Dibujo', color_actual[2], 255, actualizar_color)
cv2.createTrackbar('G', 'Dibujo', color_actual[1], 255, actualizar_color)
cv2.createTrackbar('B', 'Dibujo', color_actual[0], 255, actualizar_color)

cam = cv2.VideoCapture(0)
estado_mouse = {'preview': None}

cv2.setMouseCallback('Dibujo', mouse_evento, estado_mouse)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    color_actual[2] = cv2.getTrackbarPos('R', 'Dibujo')
    color_actual[1] = cv2.getTrackbarPos('G', 'Dibujo')
    color_actual[0] = cv2.getTrackbarPos('B', 'Dibujo')

    frame_dibujo = frame.copy()

    for figura in figuras:
        tipo, p1, p2, color = figura
        if tipo == 'r':
            cv2.rectangle(frame_dibujo, p1, p2, color, 2)
        elif tipo == 'l':
            cv2.line(frame_dibujo, p1, p2, color, 2)
        elif tipo == 'c':
            centro = p1
            radio = int(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5)
            cv2.circle(frame_dibujo, centro, radio, color, 2)

    if estado_mouse['preview'] is not None:
        p1, p2 = estado_mouse['preview']
        if tipo_figura == 'r':
            cv2.rectangle(frame_dibujo, p1, p2, tuple(color_actual), 1)
        elif tipo_figura == 'l':
            cv2.line(frame_dibujo, p1, p2, tuple(color_actual), 1)
        elif tipo_figura == 'c':
            centro = p1
            radio = int(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5)
            cv2.circle(frame_dibujo, centro, radio, tuple(color_actual), 1)

    cv2.imshow('Dibujo', frame_dibujo)

    cv2.imshow("Dibujo", frame_dibujo)

    preview_color = np.zeros((100, 100, 3), dtype=np.uint8)
    preview_color[:] = color_actual
    cv2.imshow("Color", preview_color)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla in [ord('r'), ord('l'), ord('c')]:
        tipo_figura = chr(tecla)

    preview_color = np.zeros((100, 100, 3), dtype=np.uint8)
    preview_color[:] = color_actual

    cv2.imshow("Color", preview_color)

cam.release()
cv2.destroyAllWindows()
