"""
1. Encode face images which are in face_data folder.
2. Save encoding data into face_data.pickle or face_data.yml.
"""
import glob
import pickle
import argparse
import face_recognition
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras.models import load_model
import cv2

# pylint: disable=maybe-no-member
def align_face(image):
    """
    Face alignment
    """
    landmark = face_recognition.face_landmarks(image, model="small")
    if len(landmark) > 0:
        eye_center = [
            np.mean(landmark[0]["left_eye"], axis=0).astype(int),
            np.mean(landmark[0]["right_eye"], axis=0).astype(int),
        ]
        vector = eye_center[0] - eye_center[1]
        angle = np.angle(complex(vector[0], vector[1]), deg=True)
        if abs(angle) >= 90:
            angle = 180 - abs(angle)
        pil_img = Image.fromarray(image)
        pil_img = pil_img.rotate(angle * np.sign(vector[0] * vector[1]))
        return np.array(pil_img)
    else:
        return None


def prewhiten(img):
    """
    Image whitening
    """
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1 / np.sqrt(img.size))
    white_img = (img - mean) / std_adj
    return white_img


def l2_normalize(values):
    """
    L2 nomlization
    """
    output = values / np.sqrt(
        np.maximum(np.sum(np.square(values), axis=-1, keepdims=True), 1e-10)
    )
    return output


class FacenetEncoding:
    """
    Encoding with facenet
    """

    def __init__(self, img_folder, model_path):
        self.filenames = glob.glob("{}/*/*".format(img_folder))
        self.image_size = 160
        self.detector = MTCNN()
        self.model = load_model(model_path)

    def process_image(self, img):
        """
        Process image:
        1. Get face bounding box
        2. Align Face
        3. Resize Face image
        4. Whiten image
        5. Rescale for keras model
        """
        faces = self.detector.detect_faces(img)
        if len(faces) > 0:
            (left, top, width, height) = faces[0]["box"]
            face = np.array(img[top : top + height, left : left + width])
            face = align_face(face)
            if face is not None:
                face_img = cv2.resize(face, (self.image_size, self.image_size))
                face_img = prewhiten(face_img)
                face_img = face_img[np.newaxis, :]
                return face_img
        return None

    def __call__(self):
        face_data_names = []
        face_data_encodings = []
        for img_path in self.filenames:
            name = img_path.split("/")[-2]
            print("---")
            print(name)
            img = cv2.imread(img_path)
            face_img = self.process_image(img)
            if face_img is not None:
                encoding = l2_normalize(np.concatenate(self.model.predict(face_img)))
                face_data_encodings.append(encoding)
                face_data_names.append(name)
                print("{} is encoded".format(img_path))
            else:
                print("No face detected in {}".format(img_path))

        face_data = [face_data_names, face_data_encodings]
        with open("face_data_facenet.pickle", "wb") as f_w:
            pickle.dump(face_data, f_w)
        f_w.close()


class DlibEncoding:
    """
    Encoding with dlib
    """

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
        with open("face_data_dlib.pickle", "wb") as f_w:
            pickle.dump(face_data, f_w)
        f_w.close()


class OpencvEncoding:
    """
    Encoding with opencv
    """

    def __init__(self, img_folder, xml_path):
        self.filenames = glob.glob("{}/*/*".format(img_folder))
        self.detector = cv2.CascadeClassifier(xml_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.image_size = 160

    def get_face_chip(self, image):
        """
        Crop Face image
        """
        face = self.detector.detectMultiScale(image, minSize=(100, 100))
        if len(face) > 0:
            left, top, width, height = face[0]
            return image[top : top + height, left : left + width]
        return None

    def process_image(self, image):
        """
        Process image:
        1. Crop face image
        2. Align face
        3. Resize face
        """
        face_chip = self.get_face_chip(image)
        if face_chip is not None:
            aligned_face = align_face(face_chip)
            if aligned_face is not None:
                face_img = cv2.resize(aligned_face, (self.image_size, self.image_size))
                return face_img
        return None

    def __call__(self):
        face_data_names = []
        face_data_chips = []
        for img_path in self.filenames:
            name = img_path.split("/")[-2]
            print("---")
            print(name)
            # read image in gray scale
            img = cv2.imread(img_path, flags=0)
            face_chip = self.process_image(img)
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
    """
    Parse argument
    """
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
    Encode face images and save pickle or yml file
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
