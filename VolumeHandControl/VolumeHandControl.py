import cv2 as cv
import numpy as np 
import time
import HandTrackingModule as htm
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

camWidth , camHeight = 1920, 1080

cap = cv.VideoCapture(0)
cap.set(3, camWidth)  # Set width
cap.set(4, camHeight)  # Set height
previousTime = 0
vol= 0
volBar = 400
volPer = 0

detector = htm.handDetector()

device = AudioUtilities.GetSpeakers()
interface = device.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
print(volRange)  # Print the volume range

minVol = volRange[0]
maxVol = volRange[1]


while True:
    success , img = cap.read()
    img = cv.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1 , y1 = lmList[4][1], lmList[4][2]
        x2 , y2 = lmList[8][1], lmList[8][2]

        cx , cy = (x1 + x2) //2 , (y1 + y2) // 2
        cv.circle(img, (cx,cy), 15, (255, 0, 255), cv.FILLED)

        cv.circle(img, (x1,y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2,y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # hand range 50 - 400
        # volume range -65.25 - 0.0

        vol = np.interp(length, [50,700], [minVol, maxVol])
        volBar = np.interp(length, [50,700], [400, 150])
        volPer = np.interp(length, [50,700], [0, 100])

        print(int(length),vol)
        volume.SetMasterVolumeLevel(vol, None)
        

        if length < 50:
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)

    cv.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(img, f'FPS: {int(volPer)} %', (40, 450), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
 
    currentTime = time.time() 
    if (currentTime - previousTime) != 0:
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv.imshow("Image", img)
    cv.waitKey(1)