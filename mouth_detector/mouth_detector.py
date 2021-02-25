import cv2
import numpy as np
import smbus2
import time

SLAVE_ADDRESS = 0x08
bus = smbus2.SMBus(1)

faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture('/dev/video0')
ds_factor = 0.5

mytiming = time.time()

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize = (20, 20),
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        break

    print(mouth_rects)


    if (len(mouth_rects) > 0):
        bus.write_byte(SLAVE_ADDRESS, ord('a'))
        mytiming = time.time()
    print(time.time() - mytiming)    
    if(time.time() - mytiming > 3.0):    
        bus.write_byte(SLAVE_ADDRESS, ord('s'))
        mytiming = time.time()
    
    cv2.imshow('Mouth Detector', frame)

    c = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
