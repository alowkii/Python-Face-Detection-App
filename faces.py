import cv2 as cv
import numpy as np
import pickle

front_face_cascade = cv.CascadeClassifier(
    'cascades\data\haarcascade_frontalface_alt2.xml')
profile_face_cascade = cv.CascadeClassifier(
    'cascades\data\haarcascade_profileface.xml')
plate_face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_license_plate_rus_16stages.xml')

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as file:
    main_labels = pickle.load(file)
    labels = {v:k for k,v in main_labels.items()}

capture = cv.VideoCapture(0)

while (True):
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    front_faces = front_face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    profile_faces = profile_face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in front_faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            print(id_)
            print(labels[id_])
            cv.putText(frame, labels[id_], (x,y), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
        
        cv.imwrite("This Image.png", roi_color)
        rect = cv.rectangle(frame, (x, y), (x+w, y+h),
                            (0, 255, 0), thickness=2)

    for (x, y, w, h) in profile_faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and  conf <= 85:
            print(id_)
            print(labels[id_])
            cv.putText(frame, labels[id_], (x,y), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
            
        cv.imwrite("This Image.png", roi_color)
        rect = cv.rectangle(frame, (x, y), (x+w, y+h),
                            (0, 255, 0), thickness=2)

    cv.imshow("Frame", frame)

    if cv.waitKey(20) & 0xFF == ord('\x1b'):
        break

capture.release()
cv.destroyAllWindows()
