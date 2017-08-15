import compare as cp
import sys
import os
import argparse
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import imghdr

def main(args):

	saveFace = None;
	cap = cv2.VideoCapture(0)
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()
	    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	    if len(faces) > 0:
	    	saveFace = frame
	    	break;
	    # Display the resulting frame
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	cv2.imwrite('C:/Users/USER/Desktop/facenet-RealTime/src/face_data/saveFace.jpg',frame)
	
	mypath = 'C:/Users/USER/Desktop/facenet-RealTime/src/face_data'
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	myImage = []
	for file in onlyfiles:
		isImage = None
		file = mypath + '/' + file
		isImage = imghdr.what(file)
		if isImage != None:
			myImage.append(file)

	#begin facenet
	cp.main(args,myImage);

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    #parser.add_argument('image_path', type=str, nargs='+', help='Images path to compare')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=-50) #44 default #-50 best test
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))