import cv2 as cv 
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv.VideoCapture("videos/3.mp4")
previousTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            h , w, c = img.shape
            print(id, landmark)
            cx, cy = int(landmark.x*w), int(landmark.y*h)
            cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)

    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv.imshow("Image", img)

    cv.waitKey(1)