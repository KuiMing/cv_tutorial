"""
SIFT for feature maching
"""
import argparse
import cv2
import numpy as np
import imageio

# pylint: disable=maybe-no-member

np.random.seed(23)


def resize_and_gray_image(image_path):
    """
    Resize images
    Convert images into gray scale
    """
    try:
        img = imageio.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


def sift_match_feature(train_image, query_image):
    """
    Match features of images with SIFT
    """
    img1, img1_gray = resize_and_gray_image(train_image)
    img2, img2_gray = resize_and_gray_image(query_image)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_1, m_2 in matches:
        if m_1.distance < 0.75 * m_2.distance:
            good_matches.append(m_1)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img1, flags=2)

    cv2.imshow("feature matching", img3)
    cv2.waitKey()


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--train-image",
        help="image path or url",
        type=str,
        default="https://i.imgur.com/8Imc4ax.jpg",
    )

    parser.add_argument(
        "-q",
        "--query-image",
        help="image path or url",
        type=str,
        default="https://i.imgur.com/6H0itcx.jpg",
    )

    args = parser.parse_args()
    return args


def main():
    """
    SIFT for feature maching
    """
    args = parse_args()
    sift_match_feature(args.train_image, args.query_image)


if __name__ == "__main__":
    main()
