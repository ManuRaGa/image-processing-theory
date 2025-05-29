import time
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('Iniciando Programa')
client = RemoteAPIClient()
sim = client.require('sim')

sim.loadScene('C:/Users/HUAWEI/Desktop/TRATAMIENTO/PARCIAL4/Ejercicio3.ttt')
visionSensor = sim.getObject('/visionSensor')

sim.setStepping(True)
sim.startSimulation()

while True:
    img, [resX, resY] = sim.getVisionSensorImg(visionSensor)
    img = np.frombuffer(img, np.uint8).reshape(resX, resY, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # Detectar ambos rangos de rojo (rojo puede estar en ambos extremos del círculo HSV)
    mask = cv2.inRange(hsv, lower_red1, upper_red1)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # Filtrar objetos muy pequeños
            (x, y), _ = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = 10
            cv2.circle(img, center, radius, (0, 255, 0), 2)


    cv2.imshow('Vision Sensor', img)
    cv2.imshow('Mascara', mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    sim.step()

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

cv2.destroyAllWindows()