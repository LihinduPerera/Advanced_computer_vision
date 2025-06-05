import cv2 as cv 
import mediapipe as mp
import time

class poseDetector():
    def __init__(
            self,
            static_image_mode=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ):
        self.static_image_mode = static_image_mode
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode =  self.static_image_mode,
            smooth_landmarks = self.smooth_landmarks,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence
        )

    def findPose(self, img , draw=True):
        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

            
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
    
    def findPosition(self, img , draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                h , w, c = img.shape
                # print(id, landmark)
                cx, cy = int(landmark.x*w), int(landmark.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 255), cv.FILLED)
        return lmList


def main():
    cap = cv.VideoCapture("videos/PD1.mp4")
    previousTime = 0
    detector = poseDetector()
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

# def main():
#     cap = cv.VideoCapture("videos/PD1.mp4")
#     previousTime = 0
#     detector = poseDetector()

#     while True:
#         success, img = cap.read()

#         # ✅ Make sure the frame is valid before continuing
#         if not success or img is None or img.size == 0:
#             print("Empty or invalid frame. Exiting loop.")
#             break

#         img = detector.findPose(img)
#         if img is None or img.size == 0:
#             print("Pose detection returned an invalid image.")
#             break

#         lmlist = detector.findPosition(img)

#         if len(lmlist) != 0:
#             try:
#                 # Protect against index errors if the landmark isn't available
#                 cv.circle(img, (lmlist[14][1], lmlist[14][2]), 20, (0, 0, 255), cv.FILLED)
#             except IndexError:
#                 print("Landmark 14 not found in current frame.")

#         currentTime = time.time()
#         fps = 1 / (currentTime - previousTime)
#         previousTime = currentTime

#         if img is not None and img.size > 0:
#             cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#             cv.imshow("Image", img)
#         else:
#             print("Image is empty before displaying.")
#             break

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv.destroyAllWindows()




if __name__ == "__main__":
    main()