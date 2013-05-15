#!/usr/bin/python

import numpy as np
import scipy.io
import cv2

if __name__ == "__main__":

	np.seterr (over='ignore')

	im = cv2.imread('teste4.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

	#print im.shape
	im = im.flatten(1) # same as im(:) in octave

	Theta1 = scipy.io.loadmat ('./theta1.mat')
	Theta2 = scipy.io.loadmat ('./theta2.mat')

	Theta1 = Theta1['Theta1']
	Theta2 = Theta2['Theta2']

	# sigmoid function
	#     g = 1.0 ./ (1.0 + exp(-z));	


	m=1;
	num_labels = 10;

	#print im.shape
	Exemplo = np.concatenate ([np.ones(1),im],axis=1)
	h1 = 1.0 / (1.0 + np.exp(-( np.dot(Exemplo , Theta1.transpose()) )));

	h1 = np.concatenate ([ np.ones(1), h1 ]);
	h2 = 1.0 / (1.0 + np.exp(-(  np.dot(h1 , Theta2.transpose()) )));

   	val = h2.max();
	index = h2.argmax()

	index = index + 1;
	print ("ans: %d  (index: %d) (val: %d)\n" % (index%10,index,val) );

