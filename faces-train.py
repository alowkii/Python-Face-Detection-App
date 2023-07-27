import os
import cv2 as cv
import numpy as np
from PIL import Image
import pickle

front_face_cascade = cv.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt.xml')
profile_face_cascade = cv.CascadeClassifier(
    'cascades/data/haarcascade_profileface.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()   
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            final_image = pil_image.resize((550,550))
            
            image_array = np.array(final_image, "uint8")
            front_faces = front_face_cascade.detectMultiScale(image_array)
            profile_faces = profile_face_cascade.detectMultiScale(image_array)
            
            for (x, y, w, h) in front_faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            
            for (x, y, w, h) in profile_faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_) 

with open("labels.pickle", 'wb') as file:
    pickle.dump(label_ids, file)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")