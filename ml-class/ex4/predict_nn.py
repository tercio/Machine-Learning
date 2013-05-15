#!/usr/bin/python

import numpy as np
import scipy.io
import cv2

def sigmoid (z):
	'''SIGMOID Compute sigmoid functoon
	      J = SIGMOID(z) computes the sigmoid of z.'''



	#     g = 1.0 ./ (1.0 + exp(-z));	
	return 1.0 / (1.0 + np.exp( -z ))


if __name__ == "__main__":

	np.seterr (over='ignore')

	# le uma imagem. Nesse caso deve ser uma imagem com um digito igual ao mnist de 20x20
	im = cv2.imread('teste4.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

	im = im.flatten(1) # same as im(:) in octave

	# carrega os Thetas gerados pelo treinamento da Rede Neural no Octave 
	# o valor aqui e um dict
	Theta1 = scipy.io.loadmat ('./theta1.mat')
	Theta2 = scipy.io.loadmat ('./theta2.mat')

	# converte os Thetas para ndarray
	Theta1 = Theta1['Theta1']
	Theta2 = Theta2['Theta2']

	# qtde de inputs (nesse caso so uma imagem)
	m=1;
	# camada de saida: dez unidades (digitos 0 - 9)
	num_labels = 10;

	# feed forward camada entrada -> camada escondida
	Exemplo = np.concatenate ([np.ones(1),im],axis=1)
	# sigmoid (z) : z e o valor de a* Theta' 
	h1 =  sigmoid (np.dot(Exemplo , Theta1.transpose()) )

	# feed forward camada escondida -> camada de saida
	h1 = np.concatenate ([ np.ones(1), h1 ]);
	h2 =  sigmoid ( np.dot(h1 , Theta2.transpose()) )

   	val = h2.max()
	# na lista de outputs, qual o que tem maior peso?  esse e o digito encontrado
	index = h2.argmax()

	# mostra os dados. Note que o index e incrementado, pois comecamos no digito 1 e o zero foi transformado para 10
	# para compatibilidade e maior facilidade com o Octave, que tem indice de array inicial igual a 1
	index = index + 1
	print ("ans: %d  (index: %d) (val: %f)\n" % (index%10,index,val) )


