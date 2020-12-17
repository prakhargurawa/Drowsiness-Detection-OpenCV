# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:59:16 2020

@author: prakh
"""

# extract face regions, including: Mouth,Right eyebrow,Right eye,Nose,Jaw etc
"""
Tutorial Reference: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/

The facial landmark detector implemented inside dlib produces 68 (x, y)-coordinates that map to specific facial structures. 
These 68 point mappings were obtained by training a shape predictor on the labeled iBUG 300-W dataset.

Examining the image, we can see that facial regions can be accessed via simple Python indexing 

The mouth can be accessed through points [48, 68].
The right eyebrow through points [17, 22].
The left eyebrow through points [22, 27].
The right eye using [36, 42].
The left eye with [42, 48].
The nose using [27, 35].
And the jaw via [0, 17].
"""
# import the necessary packages
from imutils import face_utils
from collections import OrderedDict
import numpy as np
import argparse
import imutils
import dlib
import cv2

"""
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])"""

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500) # resizing to have a width of 500 pixels 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting it to grayscale
# detect faces in the grayscale image
rects = detector(gray, 1)  #The first parameter to the detector  is our grayscale image , can work with colour images too
# The second parameter is the number of image pyramid layers to apply when upscaling the image prior to applying the detector

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# loop over the face parts individually
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		# clone the original image so we can draw on it, then
		# display the name of the face part on the image
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
		# loop over the subset of facial landmarks, drawing the
		# specific face part
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        # extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		# show the particular face part
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		cv2.waitKey(0)
	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)


"""
To run from Terminal:
    python face_part_detection.py --shape-predictor shape_predictor_68_face_landmarks.dat --image <path_to_image\image.jpg>
    
"""
"""
About visualize_facial_landmarks:
    image : The image that we are going to draw our facial landmark visualizations on.
    shape : The NumPy array that contains the 68 facial landmark coordinates that map to various facial parts.
    colors : A list of BGR tuples used to color-code each of the facial landmark regions.
    alpha : A parameter used to control the opacity of the overlay on the original image.
"""