# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
from collections import OrderedDict

def faceLandMarks(image):
	image_shape = image.shape
	output = np.zeros((image_shape[0],image_shape[1],image_shape[2]), np.uint8)
	mask = np.zeros((image_shape[0],image_shape[1],image_shape[2]), np.uint8)
	FACIAL_LANDMARKS_IDXS = OrderedDict([
		("mouth", (48, 68)),
		("right_eyebrow", (17, 22)),
		("left_eyebrow", (22, 27)),
		("right_eye", (36, 42)),
		("left_eye", (42, 48)),
		("nose", (27, 35)),
		("jaw", (0, 17))
	])
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	#detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

	rect = dlib.rectangle(left=0, top=0, right=image_shape[0], bottom=image_shape[1])
	shape = predictor(image,rect)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)


	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		if name =='right_eye' or name =='left_eye' or name =='mouth' or name =='left_eyebrow' or name =='right_eyebrow' or name =='nose': 
			(x,y),radius = cv2.minEnclosingCircle(np.array([shape[i:j]]))
			center = (int(x),int(y))
			radius = int(radius)
			cv2.circle(mask,center,radius,(1,1,1),-1,2)	
		# if name =='jaw':
		# 	rect = cv2.minAreaRect(np.array([shape[i:j]]))
		# 	box = cv2.boxPoints(rect)
		# 	box = np.int0(box)
		# 	cv2.drawContours(mask,[box],0,(1,1,1),-1)
		###rect
		#(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		#roi = image[y:y + h, x:x + w]
		#output[y:y + h, x:x + w,:] = image[y:y + h, x:x + w]
		#roi = imutils.resize(roi, width=20, inter=cv2.INTER_CUBIC)
		#cv2.imshow("ROI", roi)
		#cv2.waitKey(0)

		###circle
		# (x,y),radius = cv2.minEnclosingCircle(np.array([shape[i:j]]))
		# center = (int(x),int(y))
		# radius = int(radius)
		# cv2.circle(mask,center,radius,(1,1,1),-1,2)

		###rotate circle
		# ellipse = cv2.fitEllipse(np.array([shape[i:j]]))
		# cv2.ellipse(mask,ellipse,(1,1,1),-1)

		###rotate rect
		# rect = cv2.minAreaRect(np.array([shape[i:j]]))
		# box = cv2.boxPoints(rect)
		# box = np.int0(box)
		# cv2.drawContours(mask,[box],0,(1,1,1),-1)


	output = image * mask
	return output