import time
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

print('Iniciando programa de clasificación')
client = RemoteAPIClient()
sim = client.require('sim')

# Cargar escena
sim.loadScene('C:/Users/HUAWEI/Desktop/TRATAMIENTO/PARCIAL4/Ejercicio4.ttt')

# Obtener objetos
visionSensor = sim.getObject('/visionSensor')
joint = sim.getObject('/Revolute_joint')

# Activar motor y control loop desde el código
sim.setObjectInt32Param(joint, sim.jointintparam_motor_enabled, 1)
sim.setObjectInt32Param(joint, sim.jointintparam_ctrl_enabled, 1)

# Iniciar simulación paso a paso
sim.setStepping(True)
sim.startSimulation()

def detectar_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 70, 70], dtype=np.uint8)
    upper_green = np.array([80, 255, 255], dtype=np.uint8)

    lower_blue = np.array([100, 70, 70], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    color_detected = 'ninguno'
    
    if cv2.countNonZero(mask_green) > 500:
        color_detected = 'verde'
    elif cv2.countNonZero(mask_blue) > 500:
        color_detected = 'azul'
    return color_detected, mask_green, mask_blue

while True:    
    img, [resX, resY] = sim.getVisionSensorImg(visionSensor)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0)

    color_detected, mask_green, mask_blue = detectar_color(img)

    if color_detected == 'verde':
        print('Color detectado: Verde')
        sim.setJointTargetPosition(joint, -35 * np.pi / 180)
    elif color_detected == 'azul':
        print('Color detectado: Azul')
        sim.setJointTargetPosition(joint, 35 * np.pi / 180)

    cv2.imshow('Sensor de visión', img)
    cv2.imshow('Mascara Verde', mask_green)
    cv2.imshow('Mascara Azul', mask_blue)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    sim.step()

sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)

cv2.destroyAllWindows()