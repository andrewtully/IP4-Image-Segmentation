import sys
import argparse

import cv2
import numpy as np

## Parse the arguments supplied
def parse_arguments():
	args = argparse.ArgumentParser()
	args.add_argument("-i", "--image", required=True, help="Path to image")
	args.add_argument("-t", "--type", required=True, help="Type of edge segmentation")
	return vars(args.parse_args())

def region_growing(img, seed, threshold):
	"""
	A (very) simple implementation of region growing.
	Extracts a region of the input image depending on a start position and a stop condition.
	The input should be a single channel 8 bits image and the seed a pixel position (x, y).
	The threshold corresponds to the difference between outside pixel intensity and mean intensity of region.
	In case no new pixel is found, the growing stops.
	Outputs a single channel 8 bits binary (0 or 255) image. Extracted region is highlighted in white.
	"""

	try:
		dims = cv2.cv.GetSize(img)
	except TypeError:
		raise TypeError("(%s) img : IplImage expected!" % (sys._getframe().f_code.co_name))

	# img test
	if not(img.depth == cv2.cv.IPL_DEPTH_8U):
		raise TypeError("(%s) 8U image expected!" % (sys._getframe().f_code.co_name))
	elif not(img.nChannels is 1):
		raise TypeError("(%s) 1C image expected!" % (sys._getframe().f_code.co_name))
	# threshold tests
	if (not isinstance(threshold, int)) :
		raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
	elif threshold < 0:
		raise ValueError("(%s) Positive value expected!" % (sys._getframe().f_code.co_name))
	# seed tests
	if not((isinstance(seed, tuple)) and (len(seed) is 2) ) :
		raise TypeError("(%s) (x, y) variable expected!" % (sys._getframe().f_code.co_name))

	if (seed[0] or seed[1] ) < 0 :
		raise ValueError("(%s) Seed should have positive values!" % (sys._getframe().f_code.co_name))
	elif ((seed[0] > dims[0]) or (seed[1] > dims[1])):
		raise ValueError("(%s) Seed values greater than img size!" % (sys._getframe().f_code.co_name))

	reg = cv2.cv.CreateImage(dims, cv2.cv.IPL_DEPTH_8U, 1)
	cv2.cv.Zero(reg)

	#parameters
	mean_reg = float(img[seed[1], seed[0]])
	size = 1
	pix_area = dims[0]*dims[1]

	contour = [] # will be [ [[x1, y1], val1],..., [[xn, yn], valn] ]
	contour_val = []
	dist = 0
	# TODO: may be enhanced later with 8th connectivity
	orient = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
	cur_pix = [seed[0], seed[1]]

	#Spreading
	while(dist<threshold and size<pix_area):
	#adding pixels
		for j in range(4):
			#select new candidate
			temp_pix = [cur_pix[0] +orient[j][0], cur_pix[1] +orient[j][1]]

			#check if it belongs to the image
			is_in_img = dims[0]>temp_pix[0]>0 and dims[1]>temp_pix[1]>0 #returns boolean
			#candidate is taken if not already selected before
			if (is_in_img and (reg[temp_pix[1], temp_pix[0]]==0)):
				contour.append(temp_pix)
				contour_val.append(img[temp_pix[1], temp_pix[0]] )
				reg[temp_pix[1], temp_pix[0]] = 150
		#add the nearest pixel of the contour in it
		dist = abs(int(np.mean(contour_val)) - mean_reg)

		dist_list = [abs(i - mean_reg) for i in contour_val ]
		dist = min(dist_list)    #get min distance
		index = dist_list.index(min(dist_list)) #mean distance index
		size += 1 # updating region size
		reg[cur_pix[1], cur_pix[0]] = 255

		#updating mean MUST BE FLOAT
		mean_reg = (mean_reg*size + float(contour_val[index]))/(size+1)
		#updating seed
		cur_pix = contour[index]

		#removing pixel from neigborhood
		del contour[index]
		del contour_val[index]

	return reg

if __name__ == "__main__":
	args = parse_arguments()
	
	# Load the image and convert from BGR to RGB for matplotlib
	img = cv2.cv.LoadImage(args["image"], cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
	
	# Obtain the optimal threshold value
	img_grey = cv2.imread(args["image"])
	img_grey = cv2.cvtColor(img_grey, cv2.cv.CV_BGR2GRAY)
	thresh, _ = cv2.threshold(img_grey, 0, 255, cv2.cv.CV_THRESH_BINARY | cv2.cv.CV_THRESH_OTSU)
	
	types = {"region grow": region_growing(img, (456, 350), int(thresh)), 
			"split and merge": None}
	
	if (img is not None):
		region = types[args["type"]]
		
		cv2.cv.ShowImage("input", img)
		cv2.cv.ShowImage("region", region)
		cv2.cv.SaveImage("./" + args["type"] + "/" + args["image"], region)
		cv2.waitKey(0)
