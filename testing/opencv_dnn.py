import cv2
import numpy as np

# Load the pre-trained model from OpenCV
model_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_path = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

input_path = 0
cap = cv2.VideoCapture(input_path)

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

    # Iterate over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            startX, startY, endX, endY = box.astype(int)

            # Draw the bounding box and confidence on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"Confidence: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources

cv2.destroyAllWindows()
