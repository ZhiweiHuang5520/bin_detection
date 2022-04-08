'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import sys
sys.path.append("../")
from pixel_classification.generate_rgb_data import read_pixels
import pickle, os


class PixelClassifier():

  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    # get the parameter W from train.py
    self.W = np.array([[ 5.43928543, -3.86586367, -3.72879222],
                         [-4.0377136,   5.32774116, -3.6573421 ],
                         [-4.03975765, -3.68451264,  5.25645757]])

  def sigmoid(self,z):
    return 1 / (1 + np.exp(-z))

  def train(self, rate, iteration = 50):
    for it in range(iteration):
      sum_wr, sum_wg, sum_wb = np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
      for i in range(self.n):
        sum_wr += (self.y[0, i] * self.X[i,:].T * (1 - self.sigmoid(self.y[0, i] * self.X[i,:] @ self.w_r))).reshape((3,1))
        sum_wg += (self.y[1, i] * self.X[i,:].T * (1 - self.sigmoid(self.y[1, i] * self.X[i,:] @ self.w_g))).reshape((3,1))
        sum_wb += (self.y[2, i] * self.X[i,:].T * (1 - self.sigmoid(self.y[2, i] * self.X[i,:] @ self.w_b))).reshape((3,1))
      self.w_r += rate * sum_wr
      self.w_g += rate * sum_wg
      self.w_b += rate * sum_wb

    self.W = np.array([self.w_r,self.w_g,self.w_b])
    np.save('w_parameters.npy',self.W)

  def classify(self, X):
    '''
	    Classify a set of pixels into red, green, or blue

	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    self.W = self.W.reshape((3,3))
    Prob_mat = self.sigmoid(X @ self.W)
    y = Prob_mat.argmax(axis=1) + 1

    return y

