function [thebound, sig, delta, dmax, sigmamax] = thereotical_bound_func(n,W,c,d,D)
%THEORETICAL_BOUND_FUNC This function returns a theoretical bound for the
% script ExampleLogConstraints_comparison_rates. It also returns a 
% theoretically feasible value for the consensus stepsize, and some graph
% metrics involved in its computation.

% n is the number of agents 
% W is a square nonnegative matrix of dimension n (the adjacency matrix)
% c,d are vectors in R^n (problem data)
% D is a scalar in R (the bound on the optimal dual set of the problem)


% The following outputs only depends on the matrix W.
% "sig" is the consensus stepsize
% "delta" is the constant of nondegeneracy of the adjacency matrix W
% "dmax" is the maximum outdegree of the adjacency matrix W
% "sigmamax" is the maximum singular value of the Laplacian associated to W

% Construction of the Laplacian from the adjacency matrix W
Dout   = diag( W*ones(n,1) );
Lap    = Dout - W;


% maximum singular value of the Laplacian
singular_values = svd(Lap);
sigmamax = max(singular_values);

% nondegeneracy constant of the weighted adjacency matrix
delta = min(W(W~=0));

% maximum degree of the graph. 
out_degrees = sum(W~=0,2); % nonzero entries of each row
dmax = max(out_degrees);

% any number in [0,1]
delta_prime= 0.01;

% Auxiliary variable in the paper to compute C_u later on
delta_tilda = min(delta_prime, (1-delta_prime)*delta/dmax );

% Theoretical interval for the consensus stepsize
sigma_low = delta_tilda/delta;
simga_up = (1 - delta_tilda)/dmax;

if sigma_low >=simga_up
    msg = 'Infeasible consensus stepsize. There must be an error in the program';
    error(msg)

end

% the constant stepsize sig 
% needs to belong in the interval [sigma_low, sigma_up]
%sig =  1/2*(sigma_low + simga_up);
sig =  simga_up;


% Auxiliary multiplicative constant in the paper
C_u = 2^5/3^2 * 1/ (1 - (1 - delta_tilda/(4*n^2) ) );



% number of constraints
m = 1;


% bounds on sets and subgradients
bw = sqrt(n);
bd = 0;
hd = 0;

% bound on subgradients of c_i x_i
hfw = max(c);
% bound on subgradients of d_i log(1+x_i)
hgw = max(d);

hw = sqrt(n)*(hfw + D*sqrt(m)*hgw);


alphaw = 4*(bw^2 + bd^2) + 6*(hw^2 + hd^2) + hd*(3 + sig*sigmamax)*C_u*(bd + 2*hd);

% bounds on sets and subgradients
bz = sqrt(n)*D;
bmu = 0;
hmu = 0;

% n times the bound on the functions d_i log(1 + x_i)
hz = sqrt(n)*sqrt(max(d)*log(2));


alphaz = 4*(bmu^2+bz^2) + 6*(hmu^2 + hz^2) + hz*(3 + sig*sigmamax)*C_u*(bz + 2*hz);


thebound = 1/2*(alphaw + alphaz);


end

