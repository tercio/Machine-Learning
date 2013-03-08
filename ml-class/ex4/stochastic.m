%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%


%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 600;   % 25 hidden units
num_labels = 36;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

m = size(X,1)

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:400);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')


%
%  ---- Parametros de configuracao da Rede -------
%  You should also try different values of lambda
lambda = 0;
batch_size = 1000;
max_iter = 1500;
alpha = 1.5;
convergence = 0.02;

%
%  -----------------------------------------------
%

nn_params = initial_nn_params;


resto = mod(m,batch_size);
%X(end-resto+1:end,:) = [];
%y(end-resto+1:end,:) = [];

m = size (X,1) - resto 

start_time = time();
count = 1;

cost = [];

for iter=1:max_iter,
	printf ('--- iteracao: %05d\r',iter);
	fflush(stdout);

	for i=1:batch_size:m,

	    %for l=1 : num_labels,

		[_cost, nn_params_new] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X(i:i+batch_size-1,:), y(i:i+batch_size-1), lambda);


		nn_params = nn_params - nn_params_new * alpha;
	    %end

	end
	cost(count) = _cost;
	count = count + 1;
	if  mod(count,100) == 0, printf ('\nCost: %f\n',_cost); end
	if _cost <= convergence, break; end

end
printf ('\n');


% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


end_time = time();

printf ('Treinamento realizado em : %6.2f mins\n\n',(end_time - start_time) / 60);

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

%fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end));

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


