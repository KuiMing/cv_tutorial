import numpy as np
import cv2


def main():
    cap = cv2.VideoCapture("race.mp4")
    fgbg = cv2.createBackgroundSubtractorKNN()
    while True:
        ret, frame = cap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = fgmask / 255
            for i in range(3):
                frame[:, :, i] = frame[:, :, i] * fgmask
            cv2.imshow("frame", frame)
        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
