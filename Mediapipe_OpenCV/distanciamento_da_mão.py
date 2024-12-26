import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
import mediapipe as mp


#A biblioteca `cvzone` é uma biblioteca de alto nível que facilita
# o desenvolvimento de projetos de visão computacional usando o OpenCV 
# e o MediaPipe. Ela se destaca principalmente para aplicações 
# que envolvem detecção e rastreamento de mãos, rosto e outros objetos. 
# Principais funcionalidades do `cvzone`: 1. 
# Detecção e Rastreamento de Mãos: Facilita o uso de módulos 
# como o HandTrackingModule, simplificando a detecção de posições das mãos e dedos,
# além do rastreamento de gestos.

#cvzone==1.5.6

#cv2.CAP_DSHOW deixa a imagem mais rapida
video = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

whith = 3
hight = 4
video.set(whith, 1280)
video.set(hight, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
mp_hand = mp.solutions.hands
Hand = mp_hand.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

distPixels = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
distCM = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coef = np.polyfit(distPixels,distCM,2)

while  True:
    check,img = video.read()
    # Inverter a imagem horizontalmente (efeito espelho)
    img = cv2.flip(img, 1)
    
    hands = detector.findHands(img, draw=False)
    
    if hands:
        
        lmList = hands[0]['lmList']
        x1,y1,_ = lmList[5]
        x2,y2,_ = lmList[17]
        
        dist_px = (abs(x2-x1))
        A,B,C = coef
        #função quadratica:
        dist_cm = round((A*dist_px**2) + (B * dist_px) + C, 1)    
        
        x,y,w,h = hands[0]['bbox']
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255), 3)
        cvzone.putTextRect(img,f'{int(dist_cm)} cm',(x+5,y-10)) 
        
        print(dist_px, dist_cm)
            
    cv2.imshow('img', img)
    cv2.waitKey(1)