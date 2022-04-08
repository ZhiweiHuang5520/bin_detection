'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import os, cv2
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

class BinDetector():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		# set the trained parameter
		self.W = np.array([[  48245.09350316],
 						   [-177243.15949591],
 						   [  99748.49150485]])

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		# YOUR CODE HERE
		# convert to rgb

		# W_rgb_mat = np.load('../pixel_classification/w_parameters.npy').reshape((3,3))
		# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		h, w = img.shape[0], img.shape[1]
		img = img.reshape((h*w,3))
		mask_vec = img @ self.W
		mask_img = mask_vec.reshape((h,w))
		mask_img[mask_img>=0] = 1
		mask_img[mask_img < 0] = 0
		plt.imshow(mask_img)
		plt.show()
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		# YOUR CODE HERE
		#inputs are masks
		boxes = []
		h, w = img.shape[0], img.shape[1]
		label_img = label(img)
		regions = regionprops(label_img)
		for props in regions:
			y1, x1, y2, x2 = props.bbox
			lenx = x2 - x1
			leny = y2 - y1
			# filter
			if (lenx>leny):
				continue
			if (lenx < int(w / 20) or leny < int(h / 20)):
				continue
			img_region = img[y1:y2, x1:x2]
			if (np.mean(img_region) < 0.5):
				continue
			boxes.append([x1, y1, x2, y2])
		print(boxes)

		return boxes


