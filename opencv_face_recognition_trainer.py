import cv2
import numpy as np
import os
import glob
import argparse


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_chip_name(path):

    files = glob.glob("{}/*".format(path))
    face_chip = []
    names = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        name = f.split("/")[-1].split(".")[0]
        faces = detector.detectMultiScale(img, minSize=(100, 100))
        for (x, y, w, h) in faces:
            face_chip.append(img[y : y + h, x : x + w])
            names.append(name)
    return face_chip, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="image path", type=str)
    args = parser.parse_args()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, names = get_chip_name(args.path)
    recognizer.train(faces, np.array(list(range(len(faces)))))
    for i, name in enumerate(names):
        recognizer.setLabelInfo(i, name)
    recognizer.write("face_data.yml")


if __name__ == "__main__":
    main()