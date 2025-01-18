from tkinter import *
from tkinter.font import *
import datetime
import os
import cv2 as cv
import numpy as np
import pickle
from PIL import Image 
import pickle

window = Tk()

def generate_file_name(prefix):
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return f'{prefix}_{timestamp}.png'

def RecognizeMe():
    front_face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
    profile_face_cascade = cv.CascadeClassifier('cascades\data\haarcascade_profileface.xml')

    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

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
                cv.putText(frame, labels[id_], (x,y), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv.LINE_AA)
                
            cv.imwrite("Unrecognized/"+generate_file_name("Image"), roi_color)
            rect = cv.rectangle(frame, (x, y), (x+w, y+h),
                                (0, 255, 0), thickness=2)

        for (x, y, w, h) in profile_faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 45 and  conf <= 85:
                cv.putText(frame, labels[id_], (x,y), cv.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv.LINE_AA)
                
            cv.imwrite("U   nrecognized/"+generate_file_name("Image"), roi_color)
            rect = cv.rectangle(frame, (x, y), (x+w, y+h),
                                (0, 255, 0), thickness=2)
        
        cv.imshow("Face Recognizer", frame)

        if cv.waitKey(20) & 0xFF == ord('\x1b'):
            break
    
    capture.release()
    cv.destroyAllWindows()
    
def trainMe():
    EscapeLabel = Label(window, text="You can be recognised now!", bd=1, relief=SUNKEN, anchor=E, bg="#141414", fg="#ffcc66")
    EscapeLabel.pack(fill=X, side=BOTTOM, ipady=2)
    
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
    recognizer.save("trainer.yml")

    
    
def caputreImage(fileName = None):
    if fileName == "" or fileName.lower() == "enter name here":
        EscapeLabel = Label(window, text="Enter your name!", bd=1, relief=SUNKEN, anchor=E, bg="#141414", fg="#ffcc66")
        EscapeLabel.pack(fill=X, side=BOTTOM, ipady=2)
        return
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "Images")

    condition = False

    for root, dirs, files in os.walk(image_dir):
        k = 0
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                if fileName.lower() in os.path.basename(os.path.dirname(path)).replace("-"," ").lower():
                    k += 1
                    userRoot = root
                    baseName = os.path.basename(os.path.dirname(path))
                    condition = True
        if condition:
                break

    if(k>0):
        pass
        capture = cv.VideoCapture(0)
        ret, img = capture.read()
        capture.release()
        cv.imshow("Images",img)
        img_location = os.path.join(userRoot, str(str(k+1)+".jpg"))
        cv.imwrite(img_location, img)
    else:
        userRoot = os.path.join(image_dir,fileName)
        os.makedirs(userRoot)
        capture = cv.VideoCapture(0)
        ret, img = capture.read()
        capture.release()
        cv.imshow("Images",img)
        img_location = os.path.join(userRoot, str(str(k+1)+".jpg"))
        cv.imwrite(img_location, img)

    cv.waitKey(0)
    cv.destroyAllWindows()
    
window_width = 300
window_height = 500


def button(x,y,text,bgcolor,fgcolor,cmd):
    def on_enter(e):
        thisbutton['background'] = bgcolor
        thisbutton['foreground'] = fgcolor
        
    def on_leave(e):
        thisbutton['background'] = fgcolor
        thisbutton['foreground'] = bgcolor
        
    thisbutton = Button(window, width=42, height =2, text=text,
                        fg=bgcolor,
                        bg=fgcolor,
                        border=0,
                        activebackground=bgcolor,
                        activeforeground=fgcolor,
                        command=cmd)
    thisbutton.place(x=x,y=y)

def click(event):
    dataName.config(state=NORMAL)
    dataName.delete(0,END)
    dataName.config(fg="#ffffff")

textFont = Font(
    family="Calibri",
    size=12,
    weight="bold"
)
dataName = Entry(window, width=window_width, font=textFont, bg="#141414", fg="#ffcc66")
dataName.insert(0,"Enter name here")
dataName.config(state=DISABLED)
dataName.bind("<Button-1>",click)
dataName.pack(padx=0, pady=150)

def caputre():
    caputreImage(dataName.get())
            
button(0,0,"S T A R T", "#ffcc66","#141414",RecognizeMe)
button(0,37,"T R A I N", "#25dae9", "#141414",trainMe)
button(0,74,"C A P T U R E", "#d291bc","#141414",caputre)
button(0,111,"C L O S E", "#f86263","#141414",window.destroy)


EscapeLabel = Label(window, text="Esc To EXIT", bd=1, relief=SUNKEN, anchor=E, bg="#141414", fg="#ffcc66")
EscapeLabel.pack(fill=X, side=BOTTOM, ipady=2)

def close_window(event):
    window.destroy()

window.bind("<Escape>", close_window)
window.title("FaceRTC")
window.geometry(f"{window_width}x{window_height}")
window.resizable(FALSE, FALSE)
window.configure(bg="#141414")
window.mainloop()