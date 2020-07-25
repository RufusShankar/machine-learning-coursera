function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


A=zeros(size(theta));
A=(theta.^2);
A(1,1)=A(1,1).^(0.5);
A=sum(A)-A(1,1);
B=(y'*log(sigmoid(X*theta)))+((1-y)'*log(1-sigmoid(X*theta)));

J=-((1/m)*B)+((lambda/(2*m))*A);



C=((sigmoid(X*theta))-y).*X;

grad=((1/m)*C);

for i=2:size(grad,1)
  grad(i,1)=grad(i,1)+((lambda/m)*theta(i,1));
endfor
% =============================================================

end
