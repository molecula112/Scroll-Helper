import HandDetector
import cv2

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector.HandDetector()

    while True:
        success, img = cap.read()
        img= detector.detectHand(img)
        img = detector.findPos(img)
        cv2.imshow("Scroll Helper", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()