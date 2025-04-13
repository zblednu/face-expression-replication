import os
import torch
import cv2

def extract_frames_from_avi(input_avi, output_dir, save_frames=False):
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

        face= gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (64, 64))

        face_normalized = face_resized.astype('float32') / 255.0
        tensors.append(torch.tensor(face_normalized))

        if save_frames:
            output_path = os.path.join(output_dir, f'frame_{frame_count:03d}.jpg')
            cv2.imwrite(output_path, face_resized)
        frame_count += 1

    video.release()
    return tensors


tensors = extract_frames_from_avi('/home/bob/code/ml/nn-pair-task/dataset/s4/f1eng/an1.avi', 'test-grayscaled');
