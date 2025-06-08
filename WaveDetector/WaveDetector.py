import cv2 as cv
import time
import HandTrackingModule as htm

class WaveDetector:
    def __init__(self, buffer_size=15, threshold=15):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.relative_positions = []
        self.last_wave_time = 0
        self.cooldown = 2  # seconds

    def update(self, index_x, wrist_x):
        rel_x = index_x - wrist_x  # movement relative to wrist
        self.relative_positions.append(rel_x)

        if len(self.relative_positions) > self.buffer_size:
            self.relative_positions.pop(0)

    def detect_wave(self):
        if len(self.relative_positions) < self.buffer_size:
            return False

        direction_changes = 0
        last_direction = 0

        for i in range(1, len(self.relative_positions)):
            delta = self.relative_positions[i] - self.relative_positions[i - 1]

            if abs(delta) < self.threshold:
                continue  # Ignore small movements

            direction = 1 if delta > 0 else -1

            if last_direction != 0 and direction != last_direction:
                direction_changes += 1

            last_direction = direction

        current_time = time.time()
        if direction_changes >= 2 and (current_time - self.last_wave_time) > self.cooldown:
            self.last_wave_time = current_time
            return True

        return False

def main():
    cap = cv.VideoCapture(0)
    detector = htm.handDetector()
    waveDetector = WaveDetector()

    while True:
        success, img = cap.read()
        img = cv.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            wrist_x = lmList[0][1]     # wrist
            index_x = lmList[8][1]     # index finger tip

            waveDetector.update(index_x, wrist_x)

            if waveDetector.detect_wave():
                print("Hi 👋")
                cv.putText(img, "Hi 👋", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
