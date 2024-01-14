import cv2
import numpy

# Create a VideoCapture object for your webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error opening webcam")
    exit()

f1 = open("drowsinessDetectionPy/eyeMatrix.txt", "w")
f2 = open("drowsinessDetectionPy/eyeMatrixAvg.txt", "w")

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
            eye_cascade = cv2.CascadeClassifier('drowsinessDetectionPy/haarcascades/haarcascade_lefteye_2splits.xml')
            gray_detected_eyes = eye_cascade.detectMultiScale(gray_face_region)
            if len(gray_detected_eyes) > 0 :
                for (xe, ye, we, he) in gray_detected_eyes:
                    numpy.savetxt(f1, gray_face_region[ye:ye+he, xe:xe+we], fmt='%d')
                    f1.write("\n\n\n")
                    f2.write(f"{numpy.average(gray_face_region[ye:ye+he, xe:xe+we])}\n\n\n")
                    cv2.imshow('Cropped Eye', gray_face_region[ye:ye+he, xe:xe+we])
            else :
                print("LEFT EYE CLOSED")
    else :
        print("NO FACE DETECTED")
    
    # Close the window with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
f1.close()
f2.close()
# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
