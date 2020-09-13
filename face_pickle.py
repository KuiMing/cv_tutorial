"""
1. Encode face images which are in face_data folder.
2. Save encoding data into face_data.pickle or face_data.yml.
"""
import glob
import pickle
import face_recognition
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras.models import load_model
import cv2
import argparse

# pylint: disable=maybe-no-member


class FacenetEncoding:
    def __init__(self, img_folder, model_path):
        self.filenames = glob.glob("{}/*/*".format(img_folder))
        self.image_size = 160
        self.detector = MTCNN()
        self.model = load_model(model_path)

    def prewhiten(self, img):
        mean = np.mean(img)
        std = np.std(img)
        std_adj = np.maximum(std, 1 / np.sqrt(img.size))
        white_img = (img - mean) / std_adj
        return white_img

    def l2_normalize(self, x):
        output = x / np.sqrt(
            np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-10)
        )
        return output

    def resize_image(self, img):
        faces = self.detector.detect_faces(img)
        if len(faces) > 0:
            (left, top, width, height) = faces[0]["box"]
            face = np.array(img[top : top + height, left : left + width])
            output = cv2.resize(face, (self.image_size, self.image_size))
            return output
        else:
            return None

    def __call__(self):
        face_data_names = []
        face_data_encodings = []
        for img_path in self.filenames:
            name = img_path.split("/")[-2]
            print("---")
            print(name)
            face = self.resize_image(cv2.imread(img_path))
            if face is not None:
                face_img = self.prewhiten(face)
                face_img = face_img[np.newaxis, :]
                encoding = self.l2_normalize(
                    np.concatenate(self.model.predict(face_img))
                )
                face_data_encodings.append(encoding)
                face_data_names.append(name)
                print("{} is encoded".format(img_path))
            else:
                print("No face detected in {}".format(img_path))
        face_data = [face_data_names, face_data_encodings]
        with open("face_data.pickle", "wb") as f_w:
            pickle.dump(face_data, f_w)
        f_w.close()


class DlibEncoding:
    def __init__(self, img_folder):
        self.filenames = glob.glob("{}/*/*".format(img_folder))

    def __call__(self):
        face_data_names = []
        face_data_encodings = []
        for img_path in self.filenames:
            name = img_path.split("/")[-2]
            print("---")
            print(name)
            image = face_recognition.load_image_file(img_path)
            face_encodings = face_recognition.face_encodings(image, num_jitters=4)
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                face_data_names.append(name)
                face_data_encodings.append(face_encoding)
                print("{} is encoded".format(img_path))
            else:
                print("No face detected in {}".format(img_path))
        face_data = [face_data_names, face_data_encodings]
        with open("face_data.pickle", "wb") as f_w:
            pickle.dump(face_data, f_w)
        f_w.close()


class OpencvEncoding:
    def __init__(self, img_folder, xml_path):
        self.filenames = glob.glob("{}/*/*".format(img_folder))
        self.detector = cv2.CascadeClassifier(xml_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def get_face_chip(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = self.detector.detectMultiScale(img, minSize=(100, 100))
        if len(face) > 0:
            left, top, width, height = face[0]
            return img[top : top + height, left : left + width]
        else:
            return None

    def __call__(self):
        face_data_names = []
        face_data_chips = []
        for img_path in self.filenames:
            name = img_path.split("/")[-2]
            print("---")
            print(name)
            face_chip = self.get_face_chip(cv2.imread(img_path))
            if face_chip is not None:
                face_data_chips.append(face_chip)
                face_data_names.append(name)
                print("{} is encoded".format(img_path))
            else:
                print("No face detected in {}".format(img_path))

        self.recognizer.train(face_data_chips, np.array(range(len(face_data_chips))))
        for i, name in enumerate(face_data_names):
            self.recognizer.setLabelInfo(i, name)
        self.recognizer.write("face_data.yml")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image folder", type=str)
    parser.add_argument(
        "-e",
        "--encoder",
        help="encoding method: dlib, facenet, opencv",
        type=str,
        default="dlib",
    )

    parser.add_argument(
        "-m", "--model", help="keras model path", type=str, default="facenet_keras.h5"
    )
    parser.add_argument(
        "-x",
        "--xml",
        help="xml path for opencv face detector",
        type=str,
        default="haarcascade_frontalface_default.xml",
    )
    args = parser.parse_args()
    return args


def main():
    """
    Encode face images and save picle or yml file
    """
    args = parse_args()
    if args.encoder == "dlib":
        encoder = DlibEncoding(img_folder=args.image)
    elif args.encoder == "facenet":
        encoder = FacenetEncoding(img_folder=args.image, model_path=args.model)
    elif args.encoder == "opencv":
        encoder = OpencvEncoding(img_folder=args.image, xml_path=args.xml)

    encoder()


if __name__ == "__main__":
    main()
