% Numerical example for the manuscript
% "Distributed saddle-point subgradient algorithms with Laplacian
% averaging"
% (C) D. Mateos-Núñez and J. Cortés, 2016


clear all
close all
clc

%%
% NUM Problem Definition
clear all

% Number of iterations
nit = 1.3 * 1e5;

% Number of agents
n = 10;

% Problem data
%c = rand(n,1); % This line must be after the random generator rng(234)
c = (1:n)';
%d = ones(n,1);
d = rand(n,1);

% TEST: take care with this parameter elsewhere
%e = ones(n,1);
e = rand(n,1);

%% 
% Description of the optimization problem

% minimize \sum_i=1^100 c_i x_i 

% subject to \sum_i=1^100 - d_i log( 1 + e_i x_i ) \le -n/10 , 
% x_i\in[0, 1] 


% Recall that n is the number of agents


% The Lagrangian is 
% L(x,mu)=\sum_i=1^100 ( c_i x_i - mu d_i log( 1 + e_i x_i ) )

% Centralized solution using "optimtool" with solver "fmincon" and
% algorithm "interior point"

x0 = zeros(n,1);

lb = zeros(1,n);
ub = ones(1,n);

MaxIter_Data = 1000;
MaxFunEvals_Data = 10000;

[x,fval,exitflag,output,lambda,grad,hessian] = test1_optimtool(x0, lb, ub, MaxFunEvals_Data, MaxIter_Data,n,c,d,e);

% The result for the problem in question is 

costfunction = fval;
output_matlab_solver = output

% Example for n=5 agents with c=1:n, d=ones(n,1), and d=ones(n,1) 
% costfunction = 0.6487312706403711

% (TEST!)
% Evaluation of objective function at the feasible point {x_i=1}
%qbar = sum(c) - sum(c(qq)'*log(2)); 

% Upper bound for the optimal dual set
%D = -2*qbar/(n/10);
D=100;

%%
% Construction of the adjacency matrix. This block uses the function 
% small(), which in turn uses the function short()

% Seed of the random generator (to have predictable outcomes)
rng(234)
% Unweighted adjacency matrix
A = smallw(n, 1, 0.1);
% Build weighted adjacency matrix W with Metropolis weights
W = zeros(n,n);
for ii = 1:n
    for jj = ii + 1:n
        if A(ii,jj) == 1
            W(ii,jj) = 1/(1 + max(sum(A(ii,:)),sum(A(:,jj))));
            W(jj,ii) = W(ii,jj);
        end
    end
end
for ii = 1:n
    W(ii,ii) = 1 - sum(W(ii,:));
end
P = W - ones(n,n)/n;
nu = norm(W - ones(n,n)/n);

%(TEST) The power makes the matrix more populated (like considering phi hop
%communication).
phi=26; %Choose phi value

% Power phi of the adjacency matrix to compute a new adjacency matrix more
% populated
Wphi = W^phi;

%%
% Distributed solution with CoBa-DD. 
% Primal Recovery from Consensus-Based Dual Decomposition for Distributed
% Convex Optimization, by
% Andrea Simonetto and Hadi Jamali-Rad


% Primal and dual variables.
xk = zeros(n, 1); 

% (Recall that each agent maintains a copy of the multiplier)
muk = zeros(n,1);

% Objective values at each iteration
fk = zeros(nit,1);

% (Sub-) gradient stepsize
eta_d = .01*n;


for t=1:nit
    
    % xktilde is the auxiliary primal variable before computing ergodic
    % sums
    xktilde = zeros(n,1);
    
    
    % Update of auxiliary primal variables.
    % Each coordinate is the minimizer of the Lagrangian
    % for a given muk.
    xktilde = ( muk .* d .* e  - c) ./ (c .* e); 
    
    % projection onto the set [0, 1]
    xktilde = max(min(xktilde ,1 ), 0);
    
    % Ergodic sum to recover the primal variable from the auxiliary primal
    % variables
    xk = xk*(t-1)/t + xktilde/t;
    
    % Update of the multipliers
    muk = Wphi*(muk + eta_d*(-d .* log(1 + e .* xktilde) + n/10/n));
    
    % Projection of multipliers onto the set [0, D]
    muk = min(max(0, muk), D); 
    
    % Update the cost
    fk(t) = c'*xk;
    
end


%%
% Distributed saddle-point subgradient algorithms with Laplacian averaging  
% by David Mateos-Núñez and Jorge Cortés

% Consensus stepsize.
sig=0.4;

% Doubly stochastic matrix coming from Laplacian
Dout   = diag( Wphi*ones(n,1) ); % This coincides with the identity
Lap    = Dout - Wphi;

% Primal and dual variables (s stands for saddle)
x_s  = zeros(n, 1);
% (Recall that each agent maintains a copy of the multiplier)
mu_s = zeros(n, 1);

% xtilde_s is the auxiliary primal variable before computing ergodic sums
xtilde_s = zeros(n, 1);

% xtilde_s_plus is the update of xtilde_s (defined for convenience)
xtilde_s_plus = zeros(n, 1);

% Objective values at each iteration
fk_s = zeros(nit, 1);



% We use the same value of D as for the CoBa-DD algorithm


% (TEST) Learning rate, as opposed to constant stepsize eta_d

lr_constant  = 1;
rate = lr_constant * learnrate_doubling_trick(nit);

% with the same rate as CoBa-DD the performance is very similar 
%rate = .01 * n * ones(1,nit);


for t=1:nit
    
    % Subgradient with respect to primal variables
    g_xtilde_s = zeros(n,1);
    
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable.
    
    g_xtilde_s = c - (mu_s .* d .* e) ./(1 + e .* xtilde_s);
    
    % Update of primal (auxiliary) variable using a (minus) subgradient
    % step
    xtilde_s_plus = xtilde_s - rate(t) * g_xtilde_s  ;
    
    % Projection step onto [0, 1]
    xtilde_s_plus = min(max(0, xtilde_s_plus), 1);
    
    % Ergodic sum to recover the primal variable from the auxiliary primal
    % variables. We also call this the running time-average
    x_s = x_s*(t-1)/t + xtilde_s/t;
    
    % Update of the multipliers
    % Here we use the Laplacian associated to the stochastic matrix Wphi
    % with design parameter "sig"
    % (This update is also different from the one in CoBa-DD)
    mu_s = ( eye(n) - sig*Lap ) * mu_s...
        + rate(t)*(-d .* log(1+ e .* xtilde_s) + n/10/n);
    
    % projection of multipliers onto the set [0, D]
    mu_s = min(max(0, mu_s), D); 
    
    %(TEST) Need to compute the time-averages for the dual variables as
    %well
    
    % Update the cost
    fk_s(t) = c'*x_s;
    
    %Update old estimate
    xtilde_s = xtilde_s_plus;
end

%%
% Simulation of Consensus-Based Primal Dual Perturbed algorithm
% Distributed Constrained Optimization by Consensus-Based Primal-Dual 
% Perturbation Method 
% by Tsung-Hui Chang and Angelia Nedic and Anna Scaglione

% (TEST)
% Gradient step sizes for primal and dual variables
eta_p = zeros(1, nit+1);
eta_p = 1 ./(0+(1: nit+1));
%eta_p= 0.1*eta_d*ones(1,nit+1);

%(TEST)
% Gradient step sizes for perturbation points
rho_1 = 0;
rho_2 = 0;

% Initialize the cummulative sum of learning rates
sum_eta_p    = zeros(1,nit+1);
sum_eta_p(1) = eta_p(1);

% Perturbatin points (p stands for "perturbed") for primal and dual
% variables
alpha_p = zeros(n,1);
beta_p  = zeros(n,1);

% Primal auxiliary variable (before computing the running weighted average)
xtilde_p = zeros(n, 1);

% xtilde_p_plus is the update of xtilde_p (defined for convenience)
xtilde_p_plus = zeros(n, 1);

% Primal and dual variables
x_p      = zeros(n, 1);
lambda_p = zeros(n, 1); % Plays the role of mu in previous algorithms

% Extra auxiliary variable that appears in the computation of beta_p
z_p = zeros(n, 1); % (This is not a multiplier estimate)

% This variables need to be initialized as the constraint evaluated at the
% primal variables
z_p = -d .* log(1+ e .* x_p) + n/10/n;



% Objective values at each iteration
fk_p = zeros(nit, 1);


for t=1:nit
    
    % Subgradient with respect to primal variables (different evaluations)
    g_xtilde_p = zeros(n,1);
    g_alpha_p  = zeros(n,1);
    
    % 1) AVERAGE CONSENSUS
    
    %(TEST)
    Wphi_eye = 1/3 * Wphi + 2/3 * eye(n);
    
    ztilde_p      =  Wphi_eye * z_p;
    
    lambdatilde_p =  Wphi_eye * lambda_p;

% (TEST) What happens if we use the Laplacian with the same stepsize?
%     ztilde_p      = ( eye(n) - sig*Lap ) * z_p;
%     
%     lambdatilde_p = ( eye(n) - sig*Lap ) * lambda_p;
    
    
    % 2) PERTURBATION POINT COMPUTATION
    % (They use gradient descent when functions are smooth)
    
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable evaluated at primal and dual variables
    g_alpha_p = c - (lambdatilde_p .* d .* e) ./(1 + e.* xtilde_p);
    
    
    % Perturbation point for primal variable
    alpha_p = xtilde_p - rho_1 * g_alpha_p;
    
    %Projection of alpha_p onto [0, 1]
    alpha_p = min(max(0, alpha_p), 1);
   
    % Perturbation point for dual variable
    beta_p = lambdatilde_p + rho_2 * n * ztilde_p;
    
    % Projections of beta_p onto [0, D]
    beta_p = min(max(0, beta_p), D);
    
    % 3) PRIMAL-DUAL PERTURBED SUBGRADIENT UPDATE
    
    % Subgradient (column vector) of Lagrangian with respect to primal
    % variable evaluated at primal variable & perturbation point for the
    % dual variable
    g_xtilde_p    = c - (beta_p .* d .* e) ./(1 + e.* xtilde_p);
    
    % Update of primal (auxiliary) variable using a (minus) subgradient
    % step
    xtilde_p_plus = xtilde_p - eta_p(t) * g_xtilde_p  ;
    
    % Projection step onto [0, 1]
    xtilde_p_plus = min(max(0, xtilde_p_plus), 1);
    
    % Update of the multipliers
    % Subgradient (column vector) of Lagrangian with respect to dual
    % variable (i.e., the constraint function) evaluated at the 
    % perturbation point for the primal variable.
    % Note that lambdatilde_p already contains the average
    lambda_p = lambdatilde_p...
        + eta_p(t) * (-d .* log(1+ e .* alpha_p) + n/10/n);
    
    
    
    % projection of multipliers onto the set [0, D]
    lambda_p = min(max(0, lambda_p), D); 
    
    % 4) AUXILIARY VARIABLES THAT APPEAR IN PERTURBATION POINT COMPUTATIONS
    
    % difference of constraint evaluated at xtilde and xtilde in previous
    % iteration
    z_p = ztilde_p + (-d .* log(1+ e .* xtilde_p_plus) + n/10/n) ...
        -(-d .* log(1+ e .* xtilde_p) + n/10/n);
 

    
    % 5) WEIGHTED RUNNING AVERAGE (can be made offline)
    
    % Update sum of learning rates
    sum_eta_p(t+1) = sum_eta_p(t) + eta_p(t+1);
    
    % Runnint weighted average to recover the primal variable from the 
    % auxiliary primal variables (Differs from CoBa-DD and CoBa-SPS in that
    % the sum is weighted by {eta_p(t)} 
    % that is square summable
    % hat{x}_t = 1/sum_eta(t) sum_{k=1}^t eta_k x_k
    
     %x_p = (sum_eta_p(t) / sum_eta_p(t+1)) * x_p... 
         + eta_p(t+1) * xtilde_p_plus;
    
    % (TEST) What happens if we compute the normal ergodic sum?
    x_p = x_p*(t-1)/t + xtilde_p/t;
    

    
    % 6) Preparing for next iteration
    
    % Update old estimate
    xtilde_p = xtilde_p_plus;
    
    % Update the cost
    fk_p(t) = c'*x_p;
    

end


%%

%Distributed solution
% Plotting
set(0, 'DefaultAxesFontSize',14)
set(0, 'DefaultTextFontSize',14)
set(0, 'DefaultTextFontName','Verdana')
set(0, 'DefaultAxesXColor',[0.3 0.3 0.3])
set(0, 'DefaultAxesYColor',[0.3 0.3 0.3])

figure(1)
% Plotting the error between the optimal value and the value of the
% iterates (this is the cost-error) as a function of the number of
% iterations
%
%loglog(abs(fk-double(costfunction)),'b','linewidth',2)
%

TT=1:nit;
%loglog(TT,abs(fk_s - double(costfunction)),'b','linewidth',2)
loglog(TT,abs(fk_s - double(costfunction)),'b',TT,abs(fk - double(costfunction)),'r',TT,abs(fk_p - double(costfunction)),'g','linewidth',2)
h_legend1=legend('CoBa-SPS','CoBa-DD','CoBa-PDP');
set(h_legend1,'interpreter','latex','fontsize',17);
% (TEST)
% axis([1 nit 1e-3 1e2])
grid on, hold on
% Plottin the function 50/x to compare the rate of convergence
%vv = [10:10:100];
%plot(vv,50./vv,'r','linewidth',2)
%xlabel('Iteration') 
xlabel('Itaration, $t$','Interpreter','Latex','FontSize',20)
ylabel('Abs. Suboptimality','Interpreter','Latex','FontSize',20)

%%

% figure(2)
% % Number of messages per iteration
% dc = (nnz(W)-n)*phi;
% % Plotting the cost-error in terms of the number of messages
% semilogx(dc*[1:nit], fk - double(costfunction), 'r','linewidth',2)
% grid on
% xlabel('Number of messages') 
% ylabel('Suboptimality')

