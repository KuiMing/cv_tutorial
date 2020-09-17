import pickle
import cv2
import face_recognition
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
from face_encoder import align_face

# pylint: disable=maybe-no-member


def label_face(frame, left, top, right, bottom, name):
    cv2.rectangle(
        frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=2
    )
    cv2.rectangle(
        frame,
        pt1=(left, bottom - 35),
        pt2=(right, bottom),
        color=(0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text=name,
        org=(left + 6, bottom - 6),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=1,
    )


class FacenetRecognition:
    def __init__(self):
        with open("face_data_facenet.pickle", "rb") as f_r:
            self.names, self.encodings = pickle.load(f_r)
        f_r.close()
        self.image_size = 160
        self.detector = MTCNN()
        self.model = load_model("facenet_keras.h5")

    def process_image(self, img):
        face = np.array(img)
        face = align_face(face)
        if face is not None:
            face_img = cv2.resize(face, (self.image_size, self.image_size))
            face_img = self.prewhiten(face_img)
            face_img = face_img[np.newaxis, :]
            return face_img

    def detect_face(self, frame, shrink):
        face_encodings = []
        face_locations = []
        small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
        faces = self.detector.detect_faces(small_frame)
        if len(faces) == 0:
            return face_locations, face_encodings
        for i in faces:
            (left, top, width, height) = i["box"]
            face_img = self.process_image(
                small_frame[top : top + height, left : left + width]
            )
            if face_img is not None:
                encoding = self.l2_normalize(
                    np.concatenate(self.model.predict(face_img))
                )
                location = (left, top, left + width, top + height)
                face_encodings.append(encoding)
                face_locations.append(location)
        return face_locations, face_encodings

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

    def recognize_face(self, frame, tolerance=0.5):
        shrink = 0.25
        face_locations, face_encodings = self.detect_face(frame, shrink)
        for location, face_encoding in zip(face_locations, face_encodings):
            left, top, right, bottom = [int(i / shrink) for i in location]
            distances = np.linalg.norm(self.encodings - face_encoding, axis=1)
            if min(distances) < tolerance:
                name = self.names[distances.argmin()]
            else:
                name = "Unknown"
            label_face(frame, left, top, right, bottom, name)


class DlibFaceRecognition:
    def __init__(self):
        with open("face_data_dlib.pickle", "rb") as f_r:
            self.names, self.encodings = pickle.load(f_r)
        f_r.close()

    def detect_face(self, frame, shrink):
        small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        return face_locations, face_encodings

    def recognize_face(self, frame, tolerance=0.5):
        shrink = 0.25
        face_locations, face_encodings = self.detect_face(frame, shrink)
        for location, face_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = [int(i / shrink) for i in location]
            distances = face_recognition.face_distance(self.encodings, face_encoding)
            if min(distances) < tolerance:
                name = self.names[distances.argmin()]
            else:
                name = "Unknown"
            label_face(frame, left, top, right, bottom, name)


class OpencvFaceRecognition:
    def __init__(self):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read("face_data.yml")
        self.face_detector = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )
        self.image_size = 160

    def detect_face(self, frame, shrink):
        small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
        faces = self.face_detector.detectMultiScale(small_frame, minSize=(100, 100))
        return faces

    def recognize_face(self, frame, tolerance=1):
        shrink = 1
        faces = self.detect_face(frame, shrink)
        for location in faces:
            left, top, width, height = [int(i / shrink) for i in location]
            right = left + width
            bottom = top + height
            img = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            aligned_face = align_face(img)
            if aligned_face is not None:
                img = cv2.resize(aligned_face, (self.image_size, self.image_size))
                label, confidence = self.face_recognizer.predict(aligned_face)
                confidence = confidence / 100
                name = self.face_recognizer.getLabelInfo(label)
                if confidence > tolerance:
                    name = "Unknown"
                label_face(frame, left, top, right, bottom, name)


RECOGNIZER = {
    "dlib": DlibFaceRecognition(),
    "facenet": FacenetRecognition(),
    "opencv": OpencvFaceRecognition(),
}


def show_face():
    cam = cv2.VideoCapture(0)
    tolerance = 0.6
    mirror = True
    recognizer_type = list(RECOGNIZER.keys())
    recognizer = RECOGNIZER["dlib"]
    switch = 0
    cv2.namedWindow("track face", cv2.WINDOW_GUI_NORMAL)
    while True:
        ret_val, frame = cam.read()
        if mirror:
            frame = cv2.flip(frame, 1)

        if ret_val:
            height, width, _ = frame.shape
            thick = int((height + width) // 900)

            timer = cv2.getTickCount()
            recognizer.recognize_face(frame, tolerance)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            fps_info = "fps: {}".format(str(int(fps)))
            tolerance_info = "tolerance: {:.2f}".format(tolerance)
            recognizer_info = "Recognizer: {}".format(recognizer_type[switch])
            info = ", ".join([fps_info, tolerance_info, recognizer_info])
            cv2.putText(
                frame,
                text=info,
                org=(10, 45),
                fontFace=0,
                fontScale=1e-3 * height,
                color=(0, 0, 255),
                thickness=thick,
            )
            cv2.putText(
                frame,
                text="m: flip image. r: switch recognizer. x and z: tune tolerance. ESC: quit.",
                org=(10, 20),
                fontFace=0,
                fontScale=1e-3 * height,
                color=(0, 0, 200),
                thickness=thick,
            )

            cv2.imshow("track face", frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        # press m to flip frame
        if chr(keyboard & 255) == "m":
            mirror = not mirror
        # press r to switch methods of face recognition
        if chr(keyboard & 255) == "r":
            switch += 1
            switch %= len(recognizer_type)
            recognizer = RECOGNIZER[recognizer_type[switch]]
        if chr(keyboard & 255) == "x":
            tolerance += 0.1
        if chr(keyboard & 255) == "z":
            tolerance -= 0.1
    cam.release()
    cv2.destroyAllWindows()


def main():
    show_face()


if __name__ == "__main__":
    main()
