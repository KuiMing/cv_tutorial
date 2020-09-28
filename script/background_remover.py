"""
Remove background in video
"""
import argparse
import cv2

# pylint: disable=maybe-no-member
def main():
    """
    Remove background in video
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="video path", type=str)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    fgbg = cv2.createBackgroundSubtractorKNN()
    while True:
        ret, frame = cap.read()
        if ret:
            fgmask = fgbg.apply(frame)
            fgmask = fgmask / 255
            fgmask = fgmask > 0
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
