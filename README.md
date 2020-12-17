# Drowsiness-Detection-OpenCV

Python application to assist drivers to alert when they feel drowsy and thus reduce number of accidents. Based on research paper : http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

## Major dependency and my configurations
* Python (Python 3.7.3)
* OpenCV (cv2 4.4.0)
* Dlib (dlib 19.21.1) 
* Numpy (numpy 1.16.1)

## How to install Dlib:
dlib has a number of state-of-the-art implementations, including: Facial landmark detection,Correlation tracking,Deep metric learning https://github.com/davisking/dlib  
References to install dlib (worked for me on windows 10): https://medium.com/@aaditya.chhabra/install-dlib-python-api-for-windows-pc-97fe35e01cd

Main command :conda install -c conda-forge dlib    
Other reference to install dlib : 
* https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
* https://www.researchgate.net/post/I-am-not-able-to-install-dlib-in-anacondapython-version-371-can-anyone-help-me-with-that
* https://www.learnopencv.com/install-opencv-3-and-dlib-on-windows-python-only/

# Basic working:
First we will try to find facial landmarks using dlib after that we will detect different part of face for example left eye,nose,jaw etc.
Then we will extend this application to continous live video stream. One we are done with these we will try to findnumber of blinks of eyes using the concept of "eye aspect ratio" or EAR which is depected as below
![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/blink_detection_plot.jpg?raw=true)

Eye aspect ratio can be defined as:

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/blink_detection_equation.png?raw=true)

Using the above equation we will notice if we are able to properly capture these six coordinates and eye closes the EAR will be close to zero and can be a case of blinking or drowsiness.
If eyes are closed (EAR < threshold) for few frames we can alert the user as this might be case of drowsiness. Here threshold = 0.3 and number of frames is 48.

## Working example:
* Example of drowsiness detection :

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/PrakharDrowsy.gif?raw=true)

* Example of capturing facial landmarks :

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/Tau1FaceDetect.PNG?raw=true)

* Example of facial part detection :

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/PrakharLeftEye.PNG?raw=true)
![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/PrakharJaw.PNG?raw=true)

* Example of capturing facial landmarks on video:

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/PrakharLandmarks.gif?raw=true)

* Example of counting eye blinks :

![alt text](https://github.com/prakhargurawa/Drowsiness-Detection-OpenCV/blob/main/media/PrakharEyeBlink.gif?raw=true)


# Reference
I have followed tutorial of pyimagesearch https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
