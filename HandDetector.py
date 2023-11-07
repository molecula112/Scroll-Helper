from pynput.mouse import Controller
import mediapipe as mp
import math
import cv2

mouse = Controller()

class HandDetector():
    def __init__(self, maxHands=1, detectionCon=0.5, trackingCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mediapipeHands = mp.solutions.hands
        self.hands = self.mediapipeHands.Hands(
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8]
        self.swipeThreshold = math.radians(30)
        self.swipeDirection = None

    def detectHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)
        if self.res.multi_hand_landmarks:
            for handLm in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mediapipeHands.HAND_CONNECTIONS)
        return img

    def findPos(self, img, draw=True):
        lmList = []
        if self.res.multi_hand_landmarks:
            hand = self.res.multi_hand_landmarks[0]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 5, 255), cv2.FILLED)

            fingerCount = 0
            if len(lmList) >= 5:
                if lmList[4][2] < lmList[3][2]:
                    fingerCount += 1
                if lmList[8][2] < lmList[6][2]:
                    fingerCount += 1

            if fingerCount == 2:
                # Проверяем свайп пальцами
                if self.swipeDirection is None:
                    finger1 = lmList[self.tipIds[0]]
                    finger2 = lmList[self.tipIds[1]]
                    dx = finger2[1] - finger1[1]
                    dy = finger2[2] - finger1[2]
                    angle = math.atan2(dy, dx)

                    if abs(angle) > self.swipeThreshold:
                        if angle > 0:
                            self.swipeDirection = "down"
                        else:
                            self.swipeDirection = "up"
                else:
                    finger1 = lmList[self.tipIds[0]]
                    finger2 = lmList[self.tipIds[1]]
                    dx = finger2[1] - finger1[1]
                    dy = finger2[2] - finger1[2]
                    angle = math.atan2(dy, dx)

                    if abs(angle) < self.swipeThreshold:
                        scroll_amount = int(abs(dy) / 20)
                        if self.swipeDirection == "down":
                            for _ in range(scroll_amount):
                                mouse.scroll(0, -1)
                        elif self.swipeDirection == "up":
                            for _ in range(scroll_amount):
                                mouse.scroll(0, 1)

        return img