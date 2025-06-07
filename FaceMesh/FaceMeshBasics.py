import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture("videos/FD2.mp4")
previousTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec, drawSpec)

            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                x,y = int(lm.x * w), int(lm.y * h)
                # print(id,x, y) 

    currentTime = time.time()
    if (currentTime - previousTime) != 0:
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv.putText(img, f'FPS:{int(fps)}' , (20,70) , cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv.imshow("Image", img)
    cv.waitKey(1)