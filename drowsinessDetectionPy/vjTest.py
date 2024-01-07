import cv2

# Create a VideoCapture object for your webcam
cap = cv2.VideoCapture(2)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

# Capture frames from the webcam
while True:
    # Read a frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Error reading frame")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load the classifier and create a cascade object for face detection
    face_cascade = cv2.CascadeClassifier('drowsinessDetectionPy/haarcascades/haarcascade_frontalface_alt.xml')
    
    detected_faces = face_cascade.detectMultiScale(gray_frame)
    
    for (column, row, width, height) in detected_faces:
        cv2.rectangle(
        frame,
        (column, row),
        (column + width, row + height),
        (0, 255, 0),
        2
        )

    # Display the grayscale frame
    cv2.imshow('Face detection', frame)

    # Close the window with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
