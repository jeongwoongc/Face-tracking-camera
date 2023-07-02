import tensorflow as tf
import cv2
import math
import numpy as np
import os
import uuid
import time
import serial
from cvzone.FaceDetectionModule import FaceDetector

# FACE_IMG_PATH = 'Facetrackingimgs_glasses'
# FACE_IMG_PATH_NG = 'Facetrackingimgs_no_glasses'

cap = cv2.VideoCapture(1)
detector = FaceDetector()
offset = 0
imgcapnum = 0
imgres = 300

# if not os.path.exists(FACE_IMG_PATH):
#     os.makedirs(FACE_IMG_PATH)
# if not os.path.exists(FACE_IMG_PATH_NG):
#     os.makedirs(FACE_IMG_PATH_NG)

arduino = serial.Serial(port='COM5', baudrate=9600)
time.sleep(2)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    if bboxs:
        center = bboxs[0]["center"]
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        
        string_centerX = str(center[0])
        string_centerY = str(center[1])

        arduino.write(chr(88).encode('utf-8'))
        arduino.write(bytes(string_centerX,'utf-8'))
        arduino.write(chr(89).encode('utf-8'))
        arduino.write(bytes(string_centerY,'utf-8'))
        
        x, y, w, h = bboxs[0]['bbox']
        img_crop = img[y + offset:y + h - offset, x + offset :x + w - offset]

        imgWhite = np.ones((imgres,imgres,3),np.uint8)*255
        
        imgCropShape = img_crop.shape

        aspectRatio = h/w

        # if aspectRatio > 1:
        #     k = imgres/h
        #     wCal = math.ceil(k*w) # ceil round up 3.5 -> 4 
        #     imgResize = cv2.resize(img_crop,(wCal, imgres))
        #     imgResizeShape = imgResize.shape
        #     wGap = math.ceil((imgres-wCal)/2)
        #     imgWhite[:, wGap:wCal+wGap] = imgResize

        # else:
        #     k = imgres/w
        #     hCal = math.ceil(k*h) # ceil round up 3.5 -> 4 
        #     imgResize = cv2.resize(img_crop,(imgres, hCal))
        #     imgResizeShape = imgResize.shape
        #     hGap = math.ceil((imgres-hCal)/2)
        #     imgWhite[hGap:hCal + hGap, :] = imgResize

        # cv2.imshow('imgwhite',img_crop)

        # key = cv2.waitKey(1)

        # if key == ord('s'):
        #     imgcapnum += 1 
        #     print('Collecting image {}'.format(imgcapnum))
        #     imgname = os.path.join(FACE_IMG_PATH,'face'+'{}.jpg'.format(str(uuid.uuid1())))
        #     cv2.imwrite(imgname, imgWhite)
        #     cv2.imshow("imgCrop", imgWhite)

        # elif key == ord('x'):
        #     imgcapnum += 1 
        #     print('Collecting image {}'.format(imgcapnum))
        #     imgname = os.path.join(FACE_IMG_PATH_NG,'face'+'{}.jpg'.format(str(uuid.uuid1())))
        #     cv2.imwrite(imgname, imgWhite)
        #     cv2.imshow("imgCrop", imgWhite)  


    cv2.imshow("Face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()