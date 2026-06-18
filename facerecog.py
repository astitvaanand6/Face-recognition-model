import cv2 as cv
import numpy as np
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['person1', 'person2']
face_recog = cv.face.LBPHFaceRecognizer_create()
face_recog.read('face_trained.yml')
capture = cv.VideoCapture(0)
while True:
    isTrue, frame = capture.read()
    if not isTrue:
        print("Could not read frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recog.predict(faces_roi)
        if confidence > 80:
            name = "Unknown"
            color = (0, 0, 255)     
        elif confidence > 60:
            name = "Uncertain"
            color = (0, 255, 255)    
        else:
            name = people[label]
            color = (0, 255, 0)      
        print(f'{name} | Confidence = {confidence:.2f}')
        cv.putText(
            frame,
            f'{name} ({confidence:.1f})',
            (x, y - 10),
            cv.FONT_HERSHEY_COMPLEX,
            0.8,
            color,
            2
        )
        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

    cv.imshow('Detected Face', frame)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()