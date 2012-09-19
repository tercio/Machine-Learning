function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

	theta = theta - (alpha/m) * ( X' * (X * theta - y) );

%	for column = 1 : size(X,2),
%	    temp(column,1) = theta(column,1) - (alpha * 1/m * sum(  (X * theta - y)'  *    X(:,column) ));
%	end;
%
%	for column = 1 : size(X,2),
%    	theta(column,1) = temp(column,1);   %simulteneous update
%	end;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    %fprintf ('iter:%d  %.4f %.4f %.4f %.4f\n',iter,theta(1,1),theta(2,1),theta(3,1),J_history(iter))

end

end
