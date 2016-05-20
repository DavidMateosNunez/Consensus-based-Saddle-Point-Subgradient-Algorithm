function [c_ineq, ceq ] = constraint_mytest(x,n,d,e)
% Inequality and equality constraint functions for the script 
% script ExampleLogConstraints.

% The variable is x in R^n
% n is a positive integer
% d and e are vectors of parameters of the same dimension as x

% Here we define the inequality constraint
% \sum_{i=1}^n -d(i) log(1 + x(i)) \le -n/10

c_ineq = 0;

for i=1:n
    c_ineq = c_ineq - d(i) * log(1 + e(i)* x(i)) + n/10/n;
end

% There are no equality constraints
ceq=[];



end

