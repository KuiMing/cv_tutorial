import pickle
import cv2
import face_recognition
import dlib

# pylint: disable=maybe-no-member


class DlibFaceRecognition:
    def __init__(self):
        with open("face_data.pickle", "rb") as f_r:
            self.names, self.encodings = pickle.load(f_r)
        f_r.close()

    def detect_face(self, frame, shrink):
        small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        return face_locations, face_encodings

    def recognize_face(self, frame, tolerance):
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


def show_face():
    cam = cv2.VideoCapture(0)
    tolerance = 0.5
    mirror = True
    recognizer = DlibFaceRecognition()
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
            # tracking_info = "Tracker: {}".format(tracking)
            info = ", ".join([fps_info, tolerance_info])
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
                text="Press m to flip image. Presss ESC to quit.",
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
    cam.release()
    cv2.destroyAllWindows()


def main():
    show_face()


if __name__ == "__main__":
    print("Press m to flip image. Presss ESC to quit.")
    main()
