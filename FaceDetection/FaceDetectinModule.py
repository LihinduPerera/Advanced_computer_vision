import cv2 as cv 
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionConfidence= 0.5):

        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection =self.mpFaceDetection.FaceDetection(minDetectionConfidence)

    def findFaces(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results =  self.faceDetection.process(imgRGB)
        # print(self.results)

        boundingBoxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                # print(detection.location_data.relative_bounding_box)
                boundingBoxClass = detection.location_data.relative_bounding_box
                h , w , c = img.shape
                boundingBox = int(boundingBoxClass.xmin * w), int(boundingBoxClass.ymin * h), \
                                int(boundingBoxClass.width * w), int(boundingBoxClass.height * h)
                
                boundingBoxes.append([id,boundingBox, detection.score])
                
                # cv.rectangle(img, boundingBox, (255,0,255), 2)
                if draw:
                    img = self.fancyDraw(img, boundingBox) #Draw the fancy box
                    
                    cv.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 20) , cv.FONT_HERSHEY_PLAIN, 2 , (255,0,255), 2) # Draw the value

        return img, boundingBoxes
    
    def fancyDraw(self, img , boundingBox , length=30 , thickness=5 , recThickness=1):
        x, y, w, h = boundingBox
        x1 , y1 = x+w, y+h

        cv.rectangle(img, boundingBox, (255,0,255), recThickness)

        # For top Left corner x, y
        cv.line(img, (x,y) , (x + length, y), (255,0,255) , thickness)
        cv.line(img, (x,y) , (x, y+length), (255,0,255) , thickness)

        # For top Right corner x1 , y
        cv.line(img, (x1,y) , (x1 - length, y), (255,0,255) , thickness)
        cv.line(img, (x1,y) , (x1, y+length), (255,0,255) , thickness)

        # For bottom Left corner x, y1
        cv.line(img, (x,y1) , (x + length, y1), (255,0,255) , thickness)
        cv.line(img, (x,y1) , (x, y1 - length), (255,0,255) , thickness)

        # For bottom Right corner x1 , y1
        cv.line(img, (x1,y1) , (x1 - length, y1), (255,0,255) , thickness)
        cv.line(img, (x1,y1) , (x1, y1 - length), (255,0,255) , thickness)

        return img

def main():
    cap = cv.VideoCapture("videos/FD1.mp4");
    previousTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, boundingBoxes = detector.findFaces(img)
        print(boundingBoxes)

        currentTime = time.time()
    # if (currentTime-previousTime) != 0:
        fps = 1/(currentTime - previousTime)
        previousTime = currentTime
        cv.putText(img, f'FPS: {int(fps)}', (20,70) , cv.FONT_HERSHEY_PLAIN, 3 , (0,255,0), 3)

        cv.imshow("Image", img)
    
        cv.waitKey(1)


if __name__ == "__main__":
    main()
