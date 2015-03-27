import argparse

import cv2
import numpy as np

## Parse the arguments supplied
def parse_arguments():
	args = argparse.ArgumentParser()
	args.add_argument("-i", "--image", required=True, help="Path to image")
	args.add_argument("-t", "--type", required=True, help="Type of edge segmentation")
	return vars(args.parse_args())

def canny(img):
	return cv2.Canny(img, 100, 200)

def prewitt(img):
	kernel_x = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
	kernel_y = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

	prewitt_x = cv2.filter2D(img, -1, kernel_x)
	prewitt_y = cv2.filter2D(img, -1, kernel_y)

	return prewitt_x + prewitt_y

def roberts(img):
	kernel_x = np.array([[1.0, 0.0], [0.0, -1.0]])
	kernel_y = np.array([[0.0, 1.0], [-1.0, 0.0]])

	roberts_x = cv2.filter2D(img, -1, kernel_x)
	roberts_y = cv2.filter2D(img, -1, kernel_y)

	return roberts_x + roberts_y

def sobel(img):
	return cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)

def frame_average(img, num_frames, types, type_of_edge):	
	avg = np.float32(img)
	weight = float(1) / num_frames
	
	for i in range(num_frames):
		edges = types[type_of_edge]
		
		cv2.accumulateWeighted(edges, avg, weight)

	return cv2.convertScaleAbs(avg)			
				
	
if __name__ == "__main__":
	args = parse_arguments()
	
	# Load the image and convert from BGR to RGB for matplotlib
	img = cv2.imread(args["image"])
	img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
	
	types = {"canny": canny(img), 
			"prewitt": prewitt(img),
			"roberts": roberts(img),
			"sobel": sobel(img)}
	
	if (img is not None):
		edges = frame_average(img, 100, types, args["type"])
		
		cv2.imshow("input", img)
		cv2.imshow("segmented_image", edges)
		cv2.imwrite("./" + args["type"] + "/" + args["image"], edges)
		cv2.waitKey(0)
