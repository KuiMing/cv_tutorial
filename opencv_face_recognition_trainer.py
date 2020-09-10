import cv2
import numpy as np
import os
import glob
import argparse


def get_chip_name(folder, xml_path):
    detector = cv2.CascadeClassifier(xml_path)
    files = glob.glob("{}/*".format(folder))
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
    parser.add_argument("-f", "--folder", help="image folder", type=str)
    parser.add_argument(
        "-x",
        "--xml",
        help="xml path",
        type=str,
        default="haarcascade_frontalface_default.xml",
    )
    args = parser.parse_args()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, names = get_chip_name(args.folder, args.xml)
    recognizer.train(faces, np.array(list(range(len(faces)))))
    for i, name in enumerate(names):
        recognizer.setLabelInfo(i, name)
    recognizer.write("face_data.yml")


if __name__ == "__main__":
    main()