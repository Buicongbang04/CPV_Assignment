''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\Data\doanh\Doanh\FPT\Summer2024\CPV301\\assignment\OpenCV-Face-Recognition\\trainer\\trainer.yml')
cascadePath = "D:\Data\doanh\Doanh\FPT\Summer2024\CPV301\\assignment\OpenCV-Face-Recognition\FacialRecognition\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Doanh', 'nghia', 'Duong'] 

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    print(faces)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        confidence = 100 - confidence
        # Check if confidence is less them 100 ==> "0" is perfect match 
        # c > 40 and w > 150
        if confidence > 40 and w < 150:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence > 44.5 and w < 180:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence > 50.6 and w < 210:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence >= 60 and w < 240:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence >= 61 and w < 255:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence >= 67 and w < 290:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        elif confidence >= 70 and w >= 290:
            id = names[id]
            confidence = "  {0}%".format(round(confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
