import os
import cv2

def vids_to_frames(path, output_dir):
    frame_count = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vid_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.avi'):
                vid_paths.append(os.path.join(root, file))

    face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for vid in vid_paths:
        label = os.path.basename(vid)[0:2]

        video = cv2.VideoCapture(vid)

        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in face_finder.detectMultiScale(gray_frame, 1.1, 4): 
                face = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (64, 64))

                output_path = os.path.join(output_dir, f'{frame_count}{label}.jpg')
                cv2.imwrite(output_path, face_resized)
                frame_count += 1

        video.release()


vids_to_frames('raw', 'processed')
