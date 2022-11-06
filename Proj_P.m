function arg = Proj_P(j,X,Prob)
    
    % Projection onto the ambiguity set of player j
    options = optimoptions(@fmincon,'Display','off');
    K = Prob.K;

    % Compute projection
    arg = fmincon(@fobj,ones(K,1),[],[],ones(1,K),1,zeros(K,1),ones(K,1),...
        @nonlcon,options);

    function obj = fobj(x)
        obj = 1/2*x'*eye(K)*x - X'*x;
    end

    function [cieq,ceq] = nonlcon(x)
        cieq = KLdiv(x,Prob.P0,K) - Prob.rho(j);
        ceq = [];
    end


end

