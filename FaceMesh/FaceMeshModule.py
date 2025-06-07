import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, static_image_mode=False, max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            # static_image_mode,
            # max_num_faces,
            # min_detection_confidence,
            # min_tracking_confidence
            max_num_faces=max_num_faces,
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img , draw=True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, 
                                           self.mpFaceMesh.FACEMESH_TESSELATION,
                                           self.drawSpec, self.drawSpec)
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x,y = int(lm.x * w), int(lm.y * h)
                    # cv.putText(img, str(id), (x,y) , cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1) #Face IDs
                    # print(id,x, y)
                    face.append([id, x, y])
                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture("videos/FD2.mp4")
    previousTime = 0
    detector = FaceMeshDetector()


    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        currentTime = time.time()
        if (currentTime - previousTime) != 0:
            fps = 1/(currentTime - previousTime)
            previousTime = currentTime
            cv.putText(img, f'FPS:{int(fps)}' , (20,70) , cv.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()