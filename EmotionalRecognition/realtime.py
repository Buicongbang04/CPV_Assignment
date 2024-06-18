import os

import cv2
import numpy as np
from keras.models import model_from_json

json_file = open(r'weight\\model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)
model.load_weights("weight\\model_emotional_recognition.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = ['angry', 'disgust', 'fear', 'happy','neutral', 'sad', 'surprise']

def extract_features(images):
    feature = np.array(images)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        image = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        pred_label = labels[np.argmax(pred)]
        cv2.putText(frame, '% s' %(pred_label), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break