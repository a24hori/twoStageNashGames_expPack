function prob = Problem(N,K,xi,r,rho0,A,b,c,alpha0,beta0,eta0,zeta0,s0)

    prob.N = N;
    prob.K = K;
    prob.xi = xi; % Suppose xi(i,k) is generated in [-1,1]

    % 1st stage information (coefficients)
    prob.matA = A;
    prob.vecb = b;
    prob.scarc = c;
    % First-stage cost function
    prob.theta = @(j,x) 1/2*prob.matA(j,j)*x(j)^2 + prob.vecb(j)*x(j) + prob.scarc(j);
    prob.dtheta = @(j,x) prob.matA(j,j)*x(j)+prob.vecb(j);
    
    % 2nd stage information
    prob.alpha0 = alpha0;
    prob.beta0 = beta0;
    prob.eta0 = eta0;
    prob.zeta0 = zeta0;
    prob.s0 = s0;
    prob.eta = @(j,xi,k) prob.eta0(j) + 0.1.*prob.xi(1,k);
    prob.zeta = @(j,xi,k) prob.zeta0(j) + 0.1.*prob.xi(1,k);
    prob.s = @(j,xi,k) prob.s0(j) + 0.1.*prob.xi(1,k);
    prob.alpha = @(xi,k) prob.alpha0 + 5.*prob.xi(2,k);
    prob.beta = @(xi,k) prob.beta0 + prob.xi(3,k);
    prob.invDemand = @(y,xi,k) prob.alpha(xi,k) - prob.beta(xi,k)*sum(y(1:N,k));
    % Second-stage cost function of player j at k-th scenario
    prob.costH = @(j,y,xi,k) 1/2*prob.eta(j,xi,k)*y(j,k)^2 + prob.zeta(j,xi,k)*y(j,k) + prob.s(j,xi,k);
    prob.gamma = @(j,y,xi,k) prob.costH(j,y,xi,k) - prob.invDemand(y,xi,k)*y(j,k);
    % TSSVI coefficient matrices and vectors
    prob.Pi = @(xi,k) diag(prob.eta(1:N,xi,k)) + prob.beta(xi,k).*(eye(N)+ones(N,N));
    prob.vecr = @(xi,k) prob.zeta(1:N,xi,k) - prob.alpha(xi,k).*ones(N,1);
    

    % Check if max_{\xi\in\Xi} p_j(\xi) > price0(j) + dprice/2 to avoid
    % infeasibility
%     if ~all(max(prob.price(xi,1:N,1:K)') > price0+dprice/2)
%         error("failed to generate feasible problem.");
%     end

    % ambiguity set (KL divergence)
    prob.P0 = 1/K.*ones(K,N);
    prob.rho = rho0;

    prob.r = r;
    
    prob.delta = 1e-3;
    prob.tol.inner = 1e-8;
    prob.tol.outer = 1e-6;

    prob.MAXITER.inner = 2000*(N-1);
    prob.MAXITER.outer = 10*(N-1);

end

