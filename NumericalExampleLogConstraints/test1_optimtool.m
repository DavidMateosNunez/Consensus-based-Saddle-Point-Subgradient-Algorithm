function [x,fval,exitflag,output,lambda,grad,hessian] = test1_optimtool(x0,lb,ub,MaxFunEvals_Data,MaxIter_Data,n,c,d,e)
%% This is an auto generated MATLAB file from Optimization Tool for the 
% script ExampleLogConstraints.

%% Start with the default options
options = optimoptions('fmincon');
%% Modify options setting
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'MaxFunEvals', MaxFunEvals_Data);
options = optimoptions(options,'MaxIter', MaxIter_Data);
[x,fval,exitflag,output,lambda,grad,hessian] = ...
fmincon(@(x)objective_mytest(x,n,c),x0,[],[],[],[],lb,ub,@(x)constraint_mytest(x,n,d,e),options);
