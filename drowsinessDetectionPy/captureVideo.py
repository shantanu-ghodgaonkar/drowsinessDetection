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

    # Display the grayscale frame
    cv2.imshow('Grayscale Webcam Feed', gray_frame)

    # Close the window with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
