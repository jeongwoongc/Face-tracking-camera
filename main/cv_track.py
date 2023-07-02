import cv2
import math
import numpy as np
import serial
import time 
import tensorflow as tf
from keras.models import load_model

# load face tracking model
facetracker = load_model('facetracker.h5')

# arduino communication
arduino = serial.Serial(port='COM5', baudrate=9600)
time.sleep(3)

def coord_send(center):

    string_centerX = str(center[0])
    string_centerY = str(center[1])

    arduino.write(chr(88).encode('utf-8'))
    arduino.write(bytes(string_centerX,'utf-8'))
    arduino.write(chr(89).encode('utf-8'))
    arduino.write(bytes(string_centerY,'utf-8'))

    return 

cap = cv2.VideoCapture(1)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (0,255,0), 2)
        
        # Configure center dot
        start_dot = np.multiply(sample_coords[:2], [450,450]).astype(int)
        end_dot = np.multiply(sample_coords[2:], [450,450]).astype(int)

        center_factor = np.divide(end_dot-start_dot, [2,2]).astype(int)
        center_coord = np.add(center_factor, start_dot).astype(int)

        cv2.circle(frame, center_coord, 5, (255, 0, 255), cv2.FILLED) 

        # Send coordinate to arduino
        coord_send(center_coord)

        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [100,0])), 
                            (0,255,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'Daniel', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)    
    
    cv2.imshow('FaceTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()