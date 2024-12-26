import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

mp_hand = mp.solutions.hands
Hand = mp_hand.Hands(max_num_hands=1) # numero maximo de mãos que eu quero que ele reconheça
mpDraw = mp.solutions.drawing_utils

while True:
    
    check,img = video.read()


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Hand.process(imgRGB)
    handsPoints = results.multi_hand_landmarks
    
    h,w,_ = img.shape
    pontos = [] # 21 pontos (21 CORDENADAS) - 0 a 20
    
    if handsPoints:
        for points in handsPoints:
            #print(points)
            
            #desenha a linha de conecção de cada ponto da mão
            mpDraw.draw_landmarks(img,points,mp_hand.HAND_CONNECTIONS)
            
            
            for id, cordenada in enumerate(points.landmark):
                cx,cy = int(cordenada.x * w), int(cordenada.y * h) 
                
                #escreve o id de cada ponto em suas cordenadas
                cv2.putText(img, str(id), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                pontos.append((cx,cy))
                
        dedos = [8,12,16,20] #dedão é o ponto 4
        dedos_levantados = 0
        
        if pontos:
            #pega a cordenada do eixo X do ponto 4 e verifica se ele esta mais
            #a esquerda do que o ponto 2, caso estege o dedo esta abaixado 
            #ISSO É PARA A MÃO ESQUERDA. PARA A MÃO DIREITA INVERTE A LOGICA
            #DO DEDÃO
            if pontos[4][0] < pontos[2][0]:
                dedos_levantados +=1
            for n in dedos:
                #pega o valor do eixo y no ponto (exemplo) 8 e verifica se ele esta mais
                #a baixo do que o ponto 6, caso estege o dedo esta abaixado 
                # o eixo y é invertido (o valor que esta mais em baixo é maior)
                if pontos[n][1] < pontos[n - 2][1]:
                    dedos_levantados +=1
                
            print(pontos[n][1], pontos[n - 2][1])
            print(dedos_levantados)

        cv2.rectangle(
            img,
            (90,20),
            (200,100),
            (255,0,0),
            -1)
        
        cv2.putText(
            img,
            str(dedos_levantados),
            (100,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            (255,255,255),
            5
        )
            
    cv2.imshow('img', img)
    cv2.waitKey(1)