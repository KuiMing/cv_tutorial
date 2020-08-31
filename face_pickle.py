"""
1. Encode face images which are in face_data folder.
2. Save encoding data into face_data.pickle.
"""
import glob
import pickle
import face_recognition
import numpy as np
from PIL import Image


def main():
    """
    Encode face images and save picle file
    """
    filenames = glob.glob('face_data/*/*')

    face_data_names = []
    face_data_encodings = []
    for img_path in filenames:
        name = img_path.split('/')[-2]
        print('---')
        print(name)
        image = face_recognition.load_image_file(img_path)
        height, width, _ = image.shape
        image_resized = np.array(
            Image.fromarray(image).resize((int(height / width * 500), 500)))
        face_encodings = face_recognition.face_encodings(image_resized,
                                                         num_jitters=4)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            face_data_names.append(name)
            face_data_encodings.append(face_encoding)
            print('{} is encoded'.format(img_path))
        else:
            print("No face detected in {}".format(img_path))
    face_data = [face_data_names, face_data_encodings]
    with open("face_data.pickle", "wb") as f_w:
        pickle.dump(face_data, f_w)
    f_w.close()

if __name__ == '__main__':
    main()
