import cv2

cam_port = 2
cam = cv2.VideoCapture(cam_port)

# reading the input using the camera
result, image = cam.read()

# If image will detected without any error,
# show result
if result:

	# showing result, it take frame name and image
	# output
	cv2.imshow("GeeksForGeeks", image)

	# saving image in local storage
	cv2.imwrite("GeeksForGeeks.png", image)

	# If keyboard interrupt occurs, destroy image
	# window
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# If captured image is corrupted, moving to else part
else:
	print("No image detected. Please! try again")


# Python program to capture a single image
# using pygame library

# importing the pygame library
# import pygame
# import pygame.camera

# # initializing the camera
# pygame.camera.init()

# # make the list of all available cameras
# camlist = pygame.camera.list_cameras()

# # if camera is detected or not
# if camlist:

# 	# initializing the cam variable with default camera
# 	cam = pygame.camera.Camera(camlist[2], (640, 480))

# 	# opening the camera
# 	cam.start()

# 	# capturing the single image
# 	image = cam.get_image()

# 	# saving the image
# 	pygame.image.save(image, "filename.jpg")

# # if camera is not detected the moving to else part
# else:
# 	print("No camera on current device")
