import os
import cv2
import pandas as pd
import numpy as np

def extract_frames_from_avi(input_avi, output_dir=False, save_frames=False):
    if save_frames and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = cv2.VideoCapture(input_avi)
    frame_count = 0
    face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    tensors = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_finder.detectMultiScale(gray_frame, 1.1, 4)[0]

        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0

        if save_frames:
            output_path = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
            cv2.imwrite(output_path, face_resized)
        frame_count += 1

        yield face_normalized

    video.release()


#tensors = extract_frames_from_avi('/home/bob/code/ml/nn-pair-task/dataset/s4/f1eng/an1.avi', 'test-grayscaled');

avi_files = []
for root, dirs, files in os.walk('dataset'):
    for file in files:
        if file.endswith('.avi'):
            avi_files.append(os.path.join(root, file))

labels = {
        'ha': 0,
        'sa': 1,
        'an': 2,
        'fe': 3,
        'su': 4,
        'di': 5,
        }

dataset = []
for avi_file in avi_files[:5]:
    basename = os.path.basename(avi_file)
    label = labels[basename[0:2]]
    print(basename, label)
    if label == None:
        raise ValueError('can\'t map label')
    generator = extract_frames_from_avi(avi_file)
    for tensor in generator:
        dataset.append([np.array(tensor).reshape(64*64), label])

df = pd.DataFrame(dataset, columns = ['tensor', 'label'])
df.to_csv('dataset.csv', index=False)
