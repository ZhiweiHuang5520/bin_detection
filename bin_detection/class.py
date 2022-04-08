import numpy as np
import os, cv2
import sys
sys.path.append("../")
from roipoly.roipoly import RoiPoly
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

class vali_classifier():
	def __init__(self):
		'''
			Initilize your stop sign detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		if os.path.exists('W_parameters.npy'):
			self.W = np.load('W_parameters.npy')
			print("Trained parameter loaded")
			folder = 'data/validation'
			self.pixels = np.zeros((1,3))
			self.y_labels = np.zeros((1,1))
			self.n_img =1
			for filename in os.listdir(folder):
				img = cv2.imread(os.path.join(folder, filename))
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

				# display the image and use roipoly for labeling
				fig, ax = plt.subplots()
				fig.suptitle('image '+str(self.n_img))
				ax.imshow(img)
				my_roi = RoiPoly(fig=fig, ax=ax, color='r')

				# get the image mask
				mask = my_roi.get_mask(img)
				positive = bool("example is positive or not(bollean): " %(input()))
				self.get_labeles(img, mask, positive)

				# display the labeled region and the image mask

				fig, (ax1, ax2) = plt.subplots(1, 2)
				fig.suptitle('%d pixels selected\n' % img[mask, :].shape[0])


				ax1.imshow(img)
				ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
				ax2.imshow(mask)

				plt.show(block=True)


				self.n_img += 1

			self.pixels = self.pixels[1:,...]
			self.y_labels = self.y_labels[1:,...]
			np.save('test_pixels.npy', self.pixels)
			np.save('test_y_labels.npy', self.y_labels)


	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))


	def get_labeles(self, img, mask, positive = True):
		if positive == False:
			mask = -1*mask
		pixels_ch = [0]*3
		for i in range(3):
			pixels_ch[i] = img[...,i].ravel()[np.flatnonzero(mask)]
			pixels_ch[i] = pixels_ch[i].reshape((len(pixels_ch[i]),1))
		pixels_rgb = np.concatenate((pixels_ch[0],pixels_ch[1],pixels_ch[2]),axis=1)
		self.pixels = np.vstack((self.pixels, pixels_rgb))
		y_p = np.ones((pixels_rgb.shape[0],1))
		if self.n_img>50:
			y_p = -1*y_p
		self.y_labels = np.concatenate((self.y_labels, y_p))


    def test(self):
        sigama_vec = self.pixels@self.W
        sigama_vec[sigama_vec>=0] = 1
        sigama_vec[sigama_vec < 0] = -1
        print('Precision: %f' % (sum(sigama_vec == self.y_labels) / self.y_labels.shape[0]))


if __name__ == '__main__':
    my_test = vali_classifier()
    my_test.test()