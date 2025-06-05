import cv2 as cv 
import mediapipe as mp
import time

cap = cv.VideoCapture("videos/FD1.mp4");
previousTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results =  faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id,detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            boundingBoxClass = detection.location_data.relative_bounding_box
            h , w , c = img.shape
            boundingBox = int(boundingBoxClass.xmin * w), int(boundingBoxClass.ymin * h), \
                            int(boundingBoxClass.width * w), int(boundingBoxClass.height * h)
            
            cv.rectangle(img, boundingBox, (255,0,255), 2)
            cv.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 20) , cv.FONT_HERSHEY_PLAIN, 2 , (255,0,255), 2)

    currentTime = time.time()
    if (currentTime-previousTime) != 0:
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv.putText(img, f'FPS: {int(fps)}', (20,70) , cv.FONT_HERSHEY_PLAIN, 3 , (0,255,0), 3)

    cv.imshow("Image", img)
    
    cv.waitKey(1)

