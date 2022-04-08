from __future__ import division
from generate_rgb_data import read_pixels
from pixel_classifier import PixelClassifier
import numpy as np

if __name__ == '__main__':
  # test the classifier

  myPixelClassifier = PixelClassifier()
  # myPixelClassifier.train(0.001, 100)

  folder = 'data/validation'
  X1_test = read_pixels(folder + '/red')
  X2_test = read_pixels(folder + '/green')
  X3_test = read_pixels(folder + '/blue')
  y1, y2, y3 = np.full(X1_test.shape[0], 1), np.full(X2_test.shape[0], 2), np.full(X3_test.shape[0], 3)
  X_text, y_test = np.concatenate((X1_test, X2_test, X3_test)), np.concatenate((y1, y2, y3))

  y_classify = myPixelClassifier.classify(X_text)
  print('Precision: %f' % (sum(y_test == y_classify) / y_test.shape[0]))

