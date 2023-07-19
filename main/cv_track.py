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
arduino = serial.Serial(port='COM5', baudrate=9600)
time.sleep(3)

def coord_send(center):
    string_centerX = str(center[0])
    string_centerY = str(center[1])

    # Prepare the data packet to send
    data_packet = f"{string_centerX},{string_centerY}\n"

    # Send the data packet to Arduino
    arduino.write(data_packet.encode())

    return

# Iterate over frames in the input stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            startX, startY, endX, endY = box.astype(int)
            boxes.append((startX, startY, endX, endY))

            # Draw the bounding box and confidence on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.circle(frame, (int((startX + endX) / 2), int((startY + endY) / 2)), 2, (0, 255, 0), 2)
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Calculate the center coordinates
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)

            # Send the center coordinates to Arduino
            coord_send((centerX, centerY))

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Read and print the data received from Arduino
    if arduino.in_waiting > 0:
        received_data = arduino.readline().decode().strip()
        print("Arduino Received:", received_data)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
