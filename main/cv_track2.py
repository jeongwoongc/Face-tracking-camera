import cv2
import math
import numpy as np
import serial
import time 
import tensorflow as tf
from keras.models import load_model

# load face tracking model
facetracker = load_model('./models/facetracker.h5')

# arduino communication
arduino = serial.Serial(port='COM5', baudrate=250000)
time.sleep(3)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    frame = cv2.flip(frame, 1)

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

        # set error coordinates
        err_x = 20*(center_coord[0] - frame.shape[1] / 2) / (frame.shape[1] / 2)
        err_y = 20*(center_coord[1] - frame.shape[0] / 2) / (frame.shape[0] / 2)
    

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
        
        arduino.write(f"{err_x:.3f}x!".encode())
        arduino.write(f"{err_y:.3f}y!".encode())
        
        print (f"X: {err_x:.3f}, Y: {err_y:.3f}")

    cv2.imshow('FaceTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()