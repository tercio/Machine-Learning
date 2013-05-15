function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% a1
a1 = X;
a1 = [ ones(m,1) a1 ];
%size(a1)
%size (Theta1)

% a2
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ ones(m,1) a2 ];
%printf ('-- size a2 %f\n',size (a2));


% a3
z3 = a2 * Theta2';
a3 = sigmoid (z3);
%printf ('-- size a3 %f\n',size (a3));

% ------------------------------------------------------------------------------------
% compute cost
% ------------------------------------------------------------------------------------

for k = 1:num_labels,

	
	%printf ('y = %f\n',size(y == k)');
	%printf ('log(a3) = %f\n',size(a3(:,k)));
	temp(k) = ( -(y == k)' * log(a3(:,k)) - (1 - (y == k)') * log (1 - a3(:,k))  );
    temp (k);
end;

J = (1/m) * sum(temp);


% ------------------------------------------------------------------------------------
% compute regularized cost function 
% ------------------------------------------------------------------------------------

vec1 = Theta1(:,2:end);
vec1 = vec1(:);
vec2 = Theta2(:,2:end);
vec2 = vec2(:);


reg1 = sum ( vec1.^2  );
reg2 = sum ( vec2.^2  );

J = J + ( (lambda/(2*m)) * (reg1 + reg2)   );

DELTA1_grad = zeros (size(Theta1));
DELTA2_grad = zeros (size(Theta2));

for t = 1:m,

	% step 1
	% feedforward pass
	a1 = X(t,:);
	a1 = [1 a1];

	%a2
	z2 = a1 * Theta1';
	a2 = sigmoid (z2);
	a2 = [1 a2];

	%a3
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);

	% step 2
	% delta for output layer
	for k = 1:num_labels,
	    %printf ('num_labels %f %f m %d y(t) %f\n',size(num_labels),t,y(t));
		delta3(k) = a3(k) - (y(t) == k);
	end;

	% step 3
	% delta for hidden layer
	delta2 = (delta3 * Theta2) .* sigmoidGradient([1 z2]);

	% step 4
	% accumulate gradient
	delta2 = delta2(2:end);
 	DELTA1_grad = DELTA1_grad + (delta2' * a1 );
 	DELTA2_grad = DELTA2_grad + (delta3' * a2 );
	

end;


Theta1_grad = (1/m) * DELTA1_grad;  
Theta2_grad = (1/m) * DELTA2_grad;  


Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ( (lambda / m) * Theta1(:,2:end) );  
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ( (lambda / m) * Theta2(:,2:end) );  

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
