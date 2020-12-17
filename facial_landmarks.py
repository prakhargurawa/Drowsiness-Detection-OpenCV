# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:53:21 2020

@author: prakh
"""

"""
Tutorial Reference: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Detecting facial landmarks is a subset of the shape prediction problem. 
Given an input image (and normally an ROI that specifies the object of interest), 
a shape predictor attempts to localize key points of interest along the shape.

Detecting facial landmarks is therefore a two step process:

Step #1: Localize the face in the image. (Using OpenCV's Harr cascades or Neural Nets)
Step #2: Detect the key facial structures on the face ROI. (example: Mouth,Left eyebrow,Nose etc)
"""

# dlib’s facial landmark detector : The pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face.
# These annotations are part of the 68 point iBUG 300-W dataset which the dlib facial landmark predictor was trained on.
# dlib facial landmark detector : 68 (x, y)-coordinates that map to facial structures on the face

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

#Utility functions
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor") # This is the path to dlib’s pre-trained facial landmark
ap.add_argument("-i", "--image", required=True,help="path to input image") #The path to the input image that we want to detect facial landmarks on.
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
rects = detector(gray, 1) #The first parameter to the detector  is our grayscale image , can work with colour images too
# The second parameter is the number of image pyramid layers to apply when upscaling the image prior to applying the detector

# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	# show the face number
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)

"""
To run from Terminal:
    python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image <path_to_image\image.jpg>
    
"""
