function [x,y,lmd] = PHA_SAA(Prob)

    N = Prob.N; % num of players
    K = Prob.K; % num of samples
    xi = Prob.xi; % data
    r = Prob.r; % regularization parameters
    %eps = 0; % Tikhonov reguralization parameter

    % initial point
    x0 = 0.5*ones(N,1);
    y0 = rand(N,K);
    lmd0 = rand(N,K);
    w0 = zeros(N,K);

    x = x0;
    y = y0;
    lmd = lmd0;
    w = w0;

    x_memo = x0;
    
    %% Define scenario dependent matrix and vector for PHA
    MAT = @(xi,k) [Prob.matA zeros(N,N) -eye(N);
                 zeros(N,N) Prob.Pi(xi,k) eye(N,N);
                 eye(N,N) -eye(N,N) zeros(N,N)]+r*eye(3*N);
    VEC = @(xi,w,x,y,lmd,k) [Prob.vecb;Prob.vecr(xi,k);zeros(N,1)]+...
                         [w(:,k)-r*x;-r*y(:,k);-r*lmd(:,k)];

    %% Define MAT and VEC for residual check (stopping criteria), original two-stage SVI
    B = [zeros(N,N) -eye(N)];
    blkD = {};
    for k=1:K
        blkD{k} = [Prob.Pi(xi,k) eye(N); -eye(N) zeros(N,N)];
    end
    q = @(xi,k) [Prob.vecr(xi,k);zeros(N,1)];
    blkq = {};
    for k=1:K
        blkq{k} = q(xi,k);
    end
    MAT_check = [Prob.matA repmat(B/K,1,K);
                repmat(-B',K,1) blkdiag(blkD{:})];
     VEC_check = [Prob.vecb;reshape(cell2mat(blkq),[],1)];

    v = [];
    for k=1:K
        v = [v; y(1:N,k); lmd(1:N,k)];
    end
    v = [x;v];

    xyl = zeros(3*N,K);
    hatx = zeros(N,K);
    ylmd = cell(K);
    
   %% ALGORITHM: Progressive Hedging by SAA (PHA-SAA)
    for i=1:Prob.MAXITER.inner
        % Check optimality
        res = norm(min(v,MAT_check*v+VEC_check));
        %fprintf("[Original] Res.: %d\n", res);
        if res < Prob.tol.outer
            fprintf("[Original] Res.: %d\n", res);
            iter = i-1;
            break;
        elseif i==Prob.MAXITER.inner
            iter = Prob.MAXITER.inner;
            warning('Algorithm exceeds MAXITER');
            fprintf("[Original] Res.: %d\n", res);
        else
            parfor k=1:K
                xyl(:,k) = pathlcp(feval(MAT,xi,k),feval(VEC,xi,w,x,y,lmd,k));           
            end
            for k=1:K
                hatx(:,k) = xyl(1:N,k);
                ylmd{k} = [xyl(N+1:2*N,k); xyl(2*N+1:3*N,k)];     
            end
            x = (hatx*ones(K,1))/K;
            for k=1:K
                y(:,k) = xyl(N+1:2*N,k);
                lmd(:,k) = xyl(2*N+1:3*N,k);
                w(:,k) = w(:,k)+r*(hatx(:,k)-x);
            end
            v = [x;reshape(cell2mat(ylmd),[],1)];
        end
    end
    % ----------------------------------------------------------------

    % check if v is the solution to the TSSVI
    v_opt = pathlcp(MAT_check,VEC_check);
    fprintf('diff. norm: %d\n', norm(v-v_opt));

    % x-y-lmd optimality
    RHS = MAT_check*v+VEC_check; % Mx+q
    compl = v'*RHS; % x'(Mx+q)
    
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
end
