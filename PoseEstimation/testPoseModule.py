import cv2 as cv 
import time
import PoseEstimationModule as pm


cap = cv.VideoCapture("videos/PD1.mp4")
# cap = cv.VideoCapture(0)
previousTime = 0

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmlist = detector.findPosition(img)
    
    if len(lmlist) != 0:
        print(lmlist[14])
        cv.circle(img, (lmlist[14][1], lmlist[14][2]), 20, (0, 0, 255), cv.FILLED) #track for a one id

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.imshow("Image", img)

    cv.waitKey(1)