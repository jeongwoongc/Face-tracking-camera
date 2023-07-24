import cv2
import numpy as np
import serial
import time

# Load the pre-trained model from OpenCV
model_path = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_path = "./models/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

input_path = 0
cap = cv2.VideoCapture(input_path)

# Arduino communication
arduino = serial.Serial(port='COM5', baudrate=250000)
time.sleep(3)

# def coord_send(center):
#     string_X = str(center[0]) + "x!"
#     string_Y = str(center[1]) + "y!"
    
#     # Send the data packet to Arduino
#     arduino.write(string_X.encode())
#     arduino.write(string_Y.encode())

#     return

# Iterate over frames in the input stream
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the network and perform forward pass
    net.setInput(blob)
    detections = net.forward()

    # Initialize the list of bounding box rectangles
    boxes = []

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            startX, startY, endX, endY = box.astype(int)
            boxes.append((startX, startY, endX, endY))

            # Draw the bounding box and confidence on the frame
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # cv2.circle(frame, (int((startX + endX) / 2), int((startY + endY) / 2)), 2, (0, 255, 0), 2)
            # text = f"Confidence: {confidence:.2f}"
            # cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate the center coordinates
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            
            err_x = 30*(centerX - frame.shape[1] / 2) / (frame.shape[1] / 2)
            err_y = 30*(centerY - frame.shape[0] / 2) / (frame.shape[0] / 2)

            # Send the center coordinates to Arduino
            arduino.write(f"{err_x:.3f}x!".encode())
            arduino.write(f"{err_y:.3f}y!".encode())
            
            print (f"X: {err_x:.3f}, Y: {err_y:.3f}")
        else:
            arduino.write("o!".encode())

    # Display the output frame
    cv2.imshow("Daniel", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
