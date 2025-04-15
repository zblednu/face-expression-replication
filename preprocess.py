import os
import cv2

def vids_to_frames(path)
    vid_paths = []
    for root, _, files in os.walk(path):
    for file in files:
        if file.endswith('.avi'):
            vid_paths.append(os.path.join(root, file))
        
    label_map = {
        'ha': 0,
        'sa': 1,
        'an': 2,
        'fe': 3,
        'su': 4,
        'di': 5,
    }

    face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for vid in vid_paths[0:50]:
    label = label_map[os.path.basename(vid)[0:2]]

    video = cv2.VideoCapture(vid)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, w, h in face_finder.detectMultiScale(gray_frame, 1.1, 4):
            face = gray_frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (64, 64))
            face_normalized = face_resized.astype('float32') / 255.0
            image_as_tensor = torch.unsqueeze(torch.tensor(face_normalized), 0) 

            print(image_as_tensor.shape)
            self.images.append(image_as_tensor)
            self.labels.append(label)

    video.release()
