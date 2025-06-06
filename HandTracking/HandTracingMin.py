import cv2 as cv 
import mediapipe as mp
import time

cap  = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()

    img = cv.flip(img, 1) # Flip the image to the correct direction

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)

                print(id,cx, cy)
                if id==0:
                    cv.circle(img, (cx,cy), 15 , (255,0,255) , cv.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv.putText(img, str(int(fps)), (10,70) , cv.FONT_HERSHEY_PLAIN, 3 , (255,0,255), 3)


    cv.imshow("Image" , img)
    cv.waitKey(1)