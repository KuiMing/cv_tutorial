"""
Grays cale image
"""
import argparse
import cv2

# pylint: disable=maybe-no-member
def main():
    """
    Grays cale image
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)

    args = parser.parse_args()

    # img = cv2.imread(args.image, flags=0)
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("img_gray.jpg", img)


if __name__ == "__main__":
    main()
