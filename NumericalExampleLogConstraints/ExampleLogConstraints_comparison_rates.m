% Numerical example for the manuscript
% "Distributed saddle-point subgradient algorithms with Laplacian
% averaging", by
% D. Mateos-Núñez and J. Cortés, 2016

% Some parts are adapted from 
% "Primal Recovery from Consensus-Based Dual Decomposition for Distributed
% Convex Optimization", by
% Andrea Simonetto and Hadi Jamali-Rad

% This script uses the following helper functions
% "test1_optimtool" (which uses "constraint_mytest" and "objective_mytest"),
% "theoretical_bound_func"
% "smallw"
% "learnrate_doubling_trick"


clear all
close all
clc

%%
%Numerical example problem definition


% Number of iterations
nit = 1.3 * 1e3;

% Number of agents
n = 50;

% Seed of the random generator. This way we generate the same "random"
% vectors in each simulation
rng(234)

% Problem data
%c = (1:n)';
c = rand(n,1);
%d = ones(n,1);
d = rand(n,1);

e = ones(n,1);
%e = rand(n,1);

% Description of the optimization problem

% minimize \sum_i=1^n c_i x_i 

% subject to \sum_i=1^n - d_i log( 1 + e_i x_i ) \le -n/10 , 
% x_i\in[0, 1] 


% Recall that n is the number of agents


% The Lagrangian is 
% L(x,mu)=\sum_i=1^n ( c_i x_i - mu d_i log( 1 + e_i x_i ) )

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


% Upper bound for the optimal dual set. Here we use the Slater vector 
% ones(n,1)
D = n * max(c) / ( log(2)*sum(d) - n/10 );

if D<=0
    msg = 'Infeasible problem. Change the random seed "rng()" or the number of agents';
    error(msg)

end

   
Bound_dual_set=D
% sum_d = sum(d)
% max_c = max(c)
    
%%
% Construction of the adjacency matrix. This block uses the function 
% small(), which in turn uses the function short()


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




% Power phi of the adjacency matrix to compute a new adjacency matrix more
% populated. 
phi=1; %Choose phi value
Wphi = W^phi;

% Some metrics for reference: average degree and maximum degree
%average_degre = 1/n*sum(sum(Wphi~=0, 2))
%max_degree = max(sum(Wphi~=0, 2))



% We conduct the entire simulation for different sequences of learning
% rates. Each run is indexed by the parameter "rr"
number_different_rates=3;

for rr=1:number_different_rates
%%
% Distributed solution with CoBa-DD. 
% "Primal Recovery from Consensus-Based Dual Decomposition for Distributed
% Convex Optimization", by
% Andrea Simonetto and Hadi Jamali-Rad


% Learning rates change with the index rr (to autormate the
% simulations)


% The first simulation uses the Doubling Trick; the rest use constant
% learning rates
if rr==1
    lr_constant  = 1;
    rate_d = lr_constant * learnrate_doubling_trick(nit);
    
else
    lr_constant  = (rr-1)^2* 0.001*n;
    rate_d = lr_constant * ones(nit,1);
    
end

% Primal variable
xk = zeros(n, 1); 

% Dual variable
muk = zeros(n,1);

% This algorithm does not require the running time-average or ergodic sum 
% of  the estimates of the dual variables. This is a test to compute the 
% saddle-point evaluation error to compare with the C-SP-SG algorithm
% muk_time_average = zeros(n,1);

% Objective values at each iteration
fk = zeros(nit,1);

% Constraint satisfaction at each iteration
cons_satisfaction_k = zeros(nit,1);

% Evaluatin of the convexconcave function (coincides with the evaluation of
% the Lagrangian under agreement of the multipliers)
lagrangian_k = zeros(nit, 1);





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
    muk = Wphi*(muk + rate_d(t)*(-d .* log(1 + e .* xktilde) + n/10/n));
    
    % Projection of multipliers onto the set [0, D]
    muk = min(max(0, muk), D);
    
    %(TEST) If we evaluate the saddle point-evaluation error, we need to
    %compute the running time-averages of the estimates of the multipliers
    %as well.
    %muk_time_average = muk_time_average*(t-1)/t + muk/t;    
    
    % Update the cost, the constraint satisfaction, and the saddle-point
    % evaluation error
    fk(t) = c'*xk;
    
    cons_satisfaction_k(t) = ones(1,n)*(-d .* log(1+ e .* xk) + n/10/n);
    
    lagrangian_k(t)= fk(t) + ones(1,n)* (muk .*...
        (-d .* log(1+ e .* xk) + n/10/n) );

    % (TEST) We checked the difference between evaluating at the estimates
    % of the multipliers or at their running time-averages
%     lagrangian_k(t)= fk(t) + ones(1,n)* (muk_time_average .*...
%         (-d .* log(1+ e .* xk) + n/10/n) );
% %     
end


%%
% Distributed solution with C-SP-SG. 
% "Distributed saddle-point subgradient algorithms with Laplacian 
% averaging", 
% by David Mateos-Núñez and Jorge Cortés

% (The following function call is not essential to run this simulation, 
% and can be commented out, adding instead the subsequent assignment for
% sig)
% "thebound" is the mutltiplicative factor of the theoretical bound
% 1/sqrt(t) for the saddle-point evaluation error. 
% "sig" is a theoretically derived consensus stepsize. 
%[thebound, sig] = thereotical_bound_func(n, Wphi, c, d, D);
[thebound, sig, delta, dmax, sigmamax] = thereotical_bound_func(n, Wphi, c, d, D)

% Typical value of consensus stepsize for a network of 50 agents
%sig=0.25;

% Weighted Laplacian
Dout   = diag( Wphi*ones(n,1) ); % This coincides with the identity
Lap    = Dout - Wphi;

% Primal and dual variables (s stands for saddle)
x_s  = zeros(n, 1);

%(Recall that each agent maintains a copy of the multiplier)
mu_s_time_average = zeros(n,1);


% xtilde_s is the auxiliary primal variable before computing ergodic sums
xtilde_s = zeros(n, 1);

% auxiliary dual variable before computing ergodic sums
mu_s = zeros(n, 1);

% xtilde_s_plus is the update of xtilde_s (defined for convenience)
xtilde_s_plus = zeros(n, 1);

% Objective values at each iteration
fk_s = zeros(nit, 1);

% Constraint satisfaction at each iteration
cons_satisfaction_s = zeros(nit,1);

% Evaluatin of the convexconcave function (coincides with the evaluation of
% the Lagrangian under agreement of the multipliers)
lagrangian_s = zeros(nit, 1);


% We use the same value of D (bound on optimal dual set) as for 
% the CoBa-DD algorithm


% (TEST) We use the same learning rates as for CoBa-DD 
% lr_constant  = 1; % defined above
% rate = lr_constant * learnrate_doubling_trick(nit);
rate = rate_d;

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
    
    %If we evaluate the saddle point-evaluation error, we need to
    %compute the running time-averages of the estimates of the multipliers
    %as well.
    mu_s_time_average = mu_s_time_average*(t-1)/t + mu_s/t;
    
    % Update the cost, constraint satisfaction, and convexconcave function
    fk_s(t) = c'*x_s;
    
    cons_satisfaction_s(t) = ones(1,n)*(-d .* log(1+ e .* x_s) + n/10/n);
    
    lagrangian_s(t)= fk_s(t) + ones(1,n)* (mu_s_time_average .*...
        (-d .* log(1+ e .* x_s) + n/10/n) );
    
    %Update old estimate
    xtilde_s = xtilde_s_plus;
end



%% 
% Plots

set(0, 'DefaultAxesFontSize',17)
set(0, 'DefaultTextFontSize',17)
set(0, 'DefaultTextFontName','Verdana')
set(0, 'DefaultAxesXColor',[0.3 0.3 0.3])
set(0, 'DefaultAxesYColor',[0.3 0.3 0.3])


% Variable to store colors for ploting later the learning rates
color=zeros(number_different_rates,3);


% Computation of colors :-) for the different runs indexed by "rr"
if rr == 1
    vcolor1 = [0 0 1];
    vcolor2 = [1 0 1];
else
    %The bigger rr (the ligher), the bigger constant learning rate 
    %(see above for the definition)
    vcolor1=  rr*2*[0.1 0.1 0.1];
    vcolor2=  rr*2*[0.1 0.1 0.1];
        
end

% Store colors to plot the learning rates
color(rr,:) = vcolor1;


% Plot of the error between the optimal value and the value of the
% iterates (this is the cost-error) as a function of the number of
% iterations
figure(1)
% Sets Latex fonts for the axis tick labels
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
TT=1:nit;
loglog(TT,abs(fk_s - double(costfunction)),'Color',vcolor1,'linewidth',1.5)
hold on
loglog(TT,abs(fk - double(costfunction)),'Color',vcolor2,'LineStyle','-.','linewidth',1.5)
h_legend1=legend('C-SP-SG','CoBa-DD');
set(h_legend1,'interpreter','latex','fontsize',17);
%xlim([1 nit])
axis([1 nit 1e-8 1])
grid on
set(gca, 'xminorgrid', 'off')
set(gca, 'yminorgrid', 'off')
hold on
xlabel('Iteration, $t$','Interpreter','Latex','FontSize',20)
%ylabel('Abs. Suboptimality','Interpreter','Latex','FontSize',20)


% Plot of the constraint satisfaction
figure(2)
% loglog(TT,cons_satisfaction_s,'b',TT,cons_satisfaction_k,'m-.','linewidth',2)
loglog(TT,cons_satisfaction_s,'Color',vcolor1,'linewidth',1.5)
hold on
loglog(TT,cons_satisfaction_k,'Color',vcolor2,'LineStyle','-.','linewidth',1.5)
h_legend2=legend('C-SP-SG','CoBa-DD');
set(h_legend2,'interpreter','latex','fontsize',17);
%xlim([1 nit])
axis([1 nit 1e-8 1])
grid on
set(gca, 'xminorgrid', 'off')
set(gca, 'yminorgrid', 'off')
hold on
xlabel('Iteration, $t$','Interpreter','Latex','FontSize',20)
%ylabel('Constraint satisfaction','Interpreter','Latex','FontSize',20)


% Plot of the saddle-point evaluation error
figure(3)
% loglog(TT,abs(lagrangian_s - double(costfunction)),'b',TT,abs(lagrangian_k- double(costfunction)),'m-.','linewidth',2)
loglog(TT,abs(lagrangian_s - double(costfunction))','Color',vcolor1,'linewidth',1.5)
hold on
loglog(TT,abs(lagrangian_k- double(costfunction))','Color',vcolor2,'LineStyle','-.','linewidth',1.5)
hold on
h_legend3=legend('C-SP-SG','CoBa-DD');
set(h_legend3,'interpreter','latex','fontsize',17);
xlim([1 nit])
%axis([1 nit 1e-3 1e2])
grid on
set(gca, 'xminorgrid', 'off')
set(gca, 'yminorgrid', 'off')
hold on
xlabel('Iteration, $t$','Interpreter','Latex','FontSize',20)
%ylabel('Saddle-point evaluation error','Interpreter','Latex','FontSize',20)



% Plot of the learning rates used in each simulation
figure(4)

loglog(TT,rate_d(1:nit),'Color',color(rr,:),'linewidth',1.5)
% h_legend4=legend('Doubling Trick','Constant $.05$','Constant .2', '$1/t$');
% set(h_legend4,'interpreter','latex','fontsize',17);
xlim([1 nit])
%axis([1 nit 1e-3 1e2])
grid on
set(gca, 'xminorgrid', 'off')
set(gca, 'yminorgrid', 'off')
hold on
xlabel('Iteration, $t$','Interpreter','Latex','FontSize',20)
%ylabel('Saddle-point evaluation error','Interpreter','Latex','FontSize',20)

end

% Add to figure 4 the curves 1/t and 1/sqrt(t)
figure(4)

loglog(TT,1./sqrt(TT),'Color','g','linewidth',1.5)
loglog(TT,1./TT,'Color','r','linewidth',1.5)
h_legend4=legend('Doubling Trick','Constant $.05$','Constant .2', '$1/\sqrt{t}$','$1/t$');
set(h_legend4,'interpreter','latex','fontsize',17);
xlim([1 nit])
%axis([1 nit 1e-3 1e2])
grid on
set(gca, 'xminorgrid', 'off')
set(gca, 'yminorgrid', 'off')
hold on
xlabel('Iteration, $t$','Interpreter','Latex','FontSize',20)
%ylabel('Saddle-point evaluation error','Interpreter','Latex','FontSize',20)





