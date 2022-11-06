function [x_best,y_best,lmd_best,p_best,state] = PHA_Alter(Prob)

    N = Prob.N;
    K = Prob.K;
    xi = Prob.xi;
    r = Prob.r;

    options_quad = optimoptions(@quadprog,'Display','off');
    options_lin  = optimoptions(@linprog,'Display','off');

    x0 = 0.5*ones(N,1);
    y0 = rand(N,K);
    lmd0 = rand(N,K);
    w0 = zeros(N,K);

    x = x0;
    y = y0;
    lmd = lmd0;
    w = w0;

    % Define coefficient matrix and vector that is used for PHA
    MAT = @(xi,k) [Prob.matA zeros(N,N) -eye(N);
                 zeros(N,N) Prob.Pi(xi,k) eye(N,N);
                 eye(N,N) -eye(N,N) zeros(N,N)]+r*eye(3*N);
    VEC = @(xi,w,x,y,lmd,k) [Prob.vecb;Prob.vecr(xi,k);zeros(N,1)]+...
                         [w(:,k)-r*x;-r*y(:,k);-r*lmd(:,k)];
        
    % Define matrix and vector for residual check
    blkD = {};
    for k=1:K
       blkD{k} = [Prob.Pi(xi,k) eye(N); -eye(N) zeros(N,N)]; 
    end
    q = @(xi,k) [Prob.vecr(xi,k);zeros(N,1)];
    blkq = {};
    for k=1:K
        blkq{k} = q(xi,k);
    end
    B = [zeros(N,N) -eye(N)];
    MAT_check = @(blk_distr_P)...
                 [Prob.matA reshape(cell2mat(blk_distr_P),[],2*N*K);
                     repmat(-B',K,1) blkdiag(blkD{:})];
    VEC_check = [Prob.vecb;reshape(cell2mat(blkq),[],1)];
    
    v = [];
    for k=1:K
        v = [v; y(1:N,k); lmd(1:N,k)];
    end
    v = [x;v];

    res_best  = inf;
    xyl = zeros(3*N,K);
    hatx = zeros(N,K);
    ylmd = cell(K);
    blk_distr_P = cell(1,K);
    P = zeros(K,N);
    state = 0;
    
    %% ALGORITHM: Alternating Progressive Hedging Alg. (A-PHA) ============
    for nu_outer = 1:Prob.MAXITER.outer
        % Maximization of the expected value of recourse function Q
        alpha = zeros(N,1);
        for j = 1:N
            recourseVec = RecourseValue(j,y,xi,K,Prob);
            KLDualObj = @(alpha) (alpha*(log(1/K*sum(exp(recourseVec./alpha))) + Prob.rho(j)));
            [alpha(j),~,exflag] = fminbnd(KLDualObj,0,1e10);
            if exflag ~= 1
                warning("A maximization step is failed.")
            end
            if alpha(j) ~= 0
                P(:,j) = Prob.P0(:,j).*exp(recourseVec./alpha(j))/(1/K*sum(exp(recourseVec./alpha(j))));
            else
                error('alpha(j) is zero.');
            end
        end
        for k=1:K
            blk_distr_P{k} = [zeros(N,N) -diag(P(k,:))];
        end
        CH_MAT = MAT_check(blk_distr_P);
        
        % Check optimality in outer loop
        % Check w.r.t. x,y,lmd
        res.outer = norm(min(v,CH_MAT*v+VEC_check));
        fprintf("[Proposed] Res.outer: %d\n", res.outer);
        if res.outer < res_best
            res_best = res.outer;
            x_best = x;
            y_best = y;
            lmd_best = lmd;
            p_best = P;
            if res.outer < Prob.tol.outer
                state = 1;
                break;
            end
        end
        if nu_outer == Prob.MAXITER.outer
            warning('Algorithm exceeds the max iter. of outer loop.');
        end

        % Solve VI (inner iteration) by PHA
        w = w0;
        for nu_inner = 1:Prob.MAXITER.inner
            % Check optimality for fixed current prob. vector
            res.inner = norm(min(v,CH_MAT*v+VEC_check));
%            fprintf("[Proposed] Res.inner: %d\n", res.inner);
            if res.inner < Prob.tol.inner
%                 iter.inner = nu_outer - 1;
                break;
            elseif nu_inner == Prob.MAXITER.inner
                warning("Algorithm exceeds MAXITER of inner loop.");
                warning("Residual (inner): %d", res.inner);
            else
                parfor k=1:K
                    % Solve LCP for each scenario
                    xyl(:,k) = pathlcp(feval(MAT,xi,k),feval(VEC,xi,w,x,y,lmd,k));
                end
                for k=1:K
                    hatx(:,k) = xyl(1:N,k);
                end
                for j = 1:N
                    x(j) = hatx(j,:)*P(:,j);
                end
                for k=1:K
                    y(:,k) = xyl(N+1:2*N,k);
                    lmd(:,k) = xyl(2*N+1:3*N,k);
                    ylmd{k} = [y(:,k); lmd(:,k)];
                    w(:,k) = w(:,k)+r*(hatx(:,k)-x);
                end
                v = [x;reshape(cell2mat(ylmd),[],1)];
            end
        end
        
    end
    % =====================================================================

    %% Check if v is the solution to TSDRVI
    v_opt = pathlcp(CH_MAT,VEC_check);
    fprintf('diff. norm: %d\n', norm(v-v_opt));

    % x-y-lmd optimality
    RHS = CH_MAT*v+VEC_check;
    compl = v'*RHS;
    if norm(v-abs(v))>Prob.delta
        fprintf('violation norm of x: %d\n', norm(v-abs(v)));
        warning('The x is not nonnegative');
    end
    if norm(RHS-abs(RHS))>Prob.delta
        fprintf('violation norm of Mx+q: %d\n', norm(RHS-abs(RHS)))
        warning('Some elements of Mx+q is negative')
    end
    if abs(compl)>Prob.delta
        fprintf('compl measure: %d\n', compl);
        warning('Complementarity is not satisfied');
    end

    %% TODO: Check if p is the solution to TSDRVI
    for j = 1:N
        % Compute projection
        if ~((Proj_P(j,P(:,j)-RecourseValue(j,y,xi,K,Prob),Prob) - P(:,j)) < 1e-8)
            error("Optimality-p is not satisfied.");
        end
    end

end
