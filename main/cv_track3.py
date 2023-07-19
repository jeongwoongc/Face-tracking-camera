import cv2

# Load the pre-trained Haar cascade classifier for face detection
cascade_path = "./models/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

input_path = 0
cap = cv2.VideoCapture(input_path)

# Iterate over frames in the input stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection using the Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Face Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()