import argparse

import cv2
import numpy as np

""" Parse the arguments supplied """
def parse_arguments():
	args = argparse.ArgumentParser()
	args.add_argument("-i", "--image", required=True, help="Path to image")
	args.add_argument("-c", "--clusters", required=True, type=int, help="Number of clusters")
	return vars(args.parse_args())
	
	
""" K-means clustering for the number of clusters specified	"""
def kmeans(img, K):
	img = cv2.GaussianBlur(img, (7,7), 0)
	
	data = img.reshape(-1,3)
	data = np.float32(data)
	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		
	ret, label, centre = cv2.kmeans(data, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	res = centre[label.flatten()]
	
	segmented_img = res.reshape((img.shape))
	return label.reshape((img.shape[0], img.shape[1])), segmented_img.astype(np.uint8)

""" Extract a particular cluster """
def extract_cluster(img, img_label, label):
	component = np.zeros(img.shape, np.uint8)
	component[img_label==label] = img[img_label==label]
	
	return component

if __name__ =="__main__":
	args = parse_arguments()
	
	# Load the image and convert from BGR to RGB for matplotlib
	img = cv2.imread(args["image"])
	
	if (img is not None):
		label, result = kmeans(img, args["clusters"])
		
		extracted = extract_cluster(img, label, 1)
		
		cv2.imshow("input", img)
		cv2.imshow("segmented_image", result)
		cv2.imshow("extracted_cluster", extracted)
		cv2.imwrite("./kmeans/" + args["image"], result)  
		cv2.waitKey(0)
		
	else:
		print "Image does not exist at " + (args["image"])
