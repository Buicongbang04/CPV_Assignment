import os

import cv2
import numpy as np
from keras.models import model_from_json
cap = cv2.VideoCapture(0)
# Face Recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer\\trainer.yml')
cascadePath = "FacialRecognition\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
names = ['None', 'Doanh', 'Nghia', 'Duong']
font = cv2.FONT_HERSHEY_SIMPLEX
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

# Emotion
json_file = open('EmotionalRecognition\\weight\\model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("EmotionalRecognition\\weight\\model_emotional_recognition.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = ['angry', 'disgust', 'fear', 'happy','neutral', 'sad', 'surprise']

def extract_features(images):
    feature = np.array(images)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5, minSize=(int(minW), int(minH)))
    for (x, y, w, h) in faces:
        image = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #  Face recognition
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        confidence = 100 - confidence
        if confidence >= 50:
            id = names[id]
        else:
            id = "unknown"

        # Emotion
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        pred_label = labels[np.argmax(pred)]

        # Text
        cv2.putText(frame, 'Emotion: %s' %(pred_label), (0, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
        cv2.putText(frame, 'Name: ' + str(id), (0, y-130), font, 1, (255, 0, 0), 2)
        cv2.putText(frame, 'Rate: ' + str(int(confidence)), (0, y-70), font, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break