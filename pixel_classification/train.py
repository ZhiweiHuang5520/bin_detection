import numpy as np
import sys
sys.path.append("../")
from pixel_classification.generate_rgb_data import read_pixels
import pickle, os


class Train_PixelClassifier():

  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    if os.path.exists('w_parameters.npy'):
      self.W = np.load('w_parameters.npy')
      print("Trained parameter loaded")
    else:
      folder = 'data/training'
      X1 = read_pixels(folder + '/red')
      X2 = read_pixels(folder + '/green')
      X3 = read_pixels(folder + '/blue')
      X = np.concatenate((X1, X2, X3))
      Y_r = np.concatenate((np.full(X1.shape[0], 1), np.full(X2.shape[0] + X3.shape[0], -1)))
      Y_g = np.concatenate((np.full(X1.shape[0], -1), np.full(X2.shape[0], 1), np.full(X3.shape[0], -1)))
      Y_b = np.concatenate((np.full(X1.shape[0] + X2.shape[0], -1), np.full(X3.shape[0], 1)))
      Y = np.array([Y_r, Y_g, Y_b])

      self.X = X
      self.n = X.shape[0]
      self.y = Y
      self.w_r, self.w_g, self.w_b = np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))
      self.rate = 0.001
      self.train( self.rate, 50)

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
    print("Training finished")

if __name__ == '__main__':
    my_training = Train_PixelClassifier()