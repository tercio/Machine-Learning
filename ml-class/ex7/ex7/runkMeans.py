#!/usr/bin/python

import cv2
import numpy as np
import scipy.io
import scipy.misc
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plotDataPoints(X, idx, K, centroids):
	#c = cm.hot(idx,1)
	plt.scatter(X[:,0], X[:,1],c=idx)
	plt.plot (centroids[:,0],centroids[:,1],markersize=18, marker='x',linestyle=' ',color='b')
	plt.show()




def computeCentroids(X, idx, K):

	(m, n) = X.shape

	centroids = np.zeros((K,n))

	for k in range (K):

		for c in range(n):
			centroids[k,c] = np.mean (X[:,c] [np.nonzero(idx == k)])

	return centroids



def findClosestCentroids(X, centroids):

	K = centroids.shape[0]

	idx = np.zeros(X.shape[0])

	for i in range(X.shape[0]):

		tmp = np.zeros(K)


		for k in range(K):
			tmp[k] = np.sum (  (   np.power(  X[i,:] - centroids[k,:] ,2)  ) )

		idx [i] = np.argmin(tmp)

	return idx



def runkMeans (X,initial_centroids,max_iters,plot_progress=False):

	# Initialize values
	(m, n) = X.shape
	K = initial_centroids.shape[0]
	centroids = initial_centroids
	previous_centroids = centroids
	idx = np.zeros(m)

	
	# Run K-Means
	for i in range(max_iters):
    
		# For each example in X, assign it to the closest centroid
		idx = findClosestCentroids(X, centroids)
    
		# Optionally, plot progress here
		if plot_progress:
			plotDataPoints(X, idx, K, centroids)
			previous_centroids = centroids
    
		# Given the memberships, compute new centroids
		centroids = computeCentroids(X, idx, K)

	return (centroids,idx)



if __name__ == "__main__":


	K = 3 
	max_iters = 3

	X = np.array([ [1,1], [1,3], [12,15], [12,12],[4,6], [6,4],[9,8],[7,8],[8,3],[2,4],[5,4],[5,6],[7,8],[8,8],[9,9],[12,12],[11,9],[9,8],[3,5],[5,8] ])

	initial_centroids = np.array ( [ [1,1 ], [5, 5], [10,10 ] ])
	
	[centroids,idx] = runkMeans (X,initial_centroids,max_iters,True)

	print centroids
	print idx
	print X

