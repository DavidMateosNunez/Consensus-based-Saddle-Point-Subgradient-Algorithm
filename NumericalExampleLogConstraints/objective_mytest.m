function f = objective_mytest(x,n,c)
% Objective function test for the script ExampleLogConstraints.

% The variable is the column vector x
% n is a positive integer (not used here)
% c is a column vector of the same dimension as x


% This function is linear
f = c'*x;

end

