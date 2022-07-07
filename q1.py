import math
from xmlrpc.client import Boolean
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")
colizaoDetectada = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image_lower_hsv = np.array([0, 50, 100])  
    image_upper_hsv = np.array([179, 255, 255])


    mask_hsv = cv2.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

    contornos, _ = cv2.findContours(mask_hsv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    mask_rgb = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2RGB) 
    contornos_img = mask_rgb.copy() # Cópia da máscara para ser desenhada "por cima"
    contornosOrdenados = sorted(contornos, key=lambda x: cv2.contourArea(x), reverse=True)

    if(len(contornosOrdenados) > 1):
        x2, y2, w2, h2 = cv2.boundingRect(contornosOrdenados[1])
        cv2.rectangle(contornos_img,(x2,y2), (x2+w2,y2+h2), (0,0,128), 5)

    x1, y1, w1, h1 = cv2.boundingRect(contornosOrdenados[0])
    cv2.rectangle(contornos_img,(x1,y1), (x1+w1,y1+h1), (0,128,0), 5)

    if(x2 > x1 and x2 < x1 + h1):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(contornos_img, "Colisao Detectada", (1000,100), font,1,(200,50,0),2,cv2.LINE_AA)
        colizaoDetectada = True
    elif(colizaoDetectada == True):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(contornos_img, "Passou a barreira", (1000,100), font,1,(200,50,0),2,cv2.LINE_AA)


    cv2.imshow("Teste",contornos_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()