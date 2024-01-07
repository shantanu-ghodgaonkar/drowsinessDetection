import cv2
import numpy

# Create a VideoCapture object for your webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

f = open("drowsinessDetectionPy/eyeMatrix.txt", "a")

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
    
    if len(detected_faces) > 0 :
    
        for (xf, yf, wf, hf) in detected_faces:
            # Crop the face region
            gray_face_region = gray_frame[yf:yf+hf, xf:xf+wf]
            eye_cascade = cv2.CascadeClassifier('drowsinessDetectionPy/haarcascades/haarcascade_eye.xml')
            gray_detected_eyes = eye_cascade.detectMultiScale(gray_face_region)
            if len(gray_detected_eyes) > 0 :
                for (xe, ye, we, he) in gray_detected_eyes:
                    numpy.savetxt(f, gray_face_region[ye:ye+he, xe:xe+we], fmt='%d')
                    cv2.imshow('Cropped Eye', gray_face_region[ye:ye+he, xe:xe+we])
            else :
                print("BOTH EYES CLOSED")
    else :
        print("NO FACE DETECTED")
    
    # Close the window with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
f.close()
# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
