"""
1. Encode face images which are in face_data folder.
2. Save encoding data into face_data.pickle.
"""
import glob
import pickle
import face_recognition
import numpy as np
from PIL import Image
from skimage.transform import resize
from mtcnn import MTCNN
from keras.models import load_model


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
            else:
                print("No face detected in {}".format(img_path))
        face_data = [face_data_names, face_data_encodings]
        with open("face_data.pickle", "wb") as f_w:
            pickle.dump(face_data, f_w)
        f_w.close()


class DlibEncoding:
    def __init__(self, img_folder, model_path):
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


def main():
    """
    Encode face images and save picle file
    """
    filenames = glob.glob("face_data/*/*")

    face_data_names = []
    face_data_encodings = []
    for img_path in filenames:
        name = img_path.split("/")[-2]
        print("---")
        print(name)
        image = face_recognition.load_image_file(img_path)
        height, width, _ = image.shape
        image_resized = np.array(
            Image.fromarray(image).resize((int(height / width * 500), 500))
        )
        face_encodings = face_recognition.face_encodings(image_resized, num_jitters=4)
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


if __name__ == "__main__":
    main()
