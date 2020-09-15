import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse

np.random.seed(23)


def resize_and_gray_image(image_path):
    try:
        img = imageio.imread(image_path)
    except FileNotFoundError:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, None, fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray


def sift_match_feature(train_image, query_image):
    img1, img1_gray = resize_and_gray_image(train_image)
    img2, img2_gray = resize_and_gray_image(query_image)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, img1, flags=2)

    plt.imshow(img3)
    plt.show()
    return good_matches


def parse_args():
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
    args = parse_args()
    matches = sift_match_feature(args.train_image, args.query_image)


if __name__ == "__main__":
    main()
