function [val] = DirectionalDerivative(vec,n,minORmax)

    % Compute the maximum of the directional derivative vec'x
    % subject to ones'*x = 0 and norm(x) = 1.

    options = optimoptions("fmincon","Display","none");
    x0 = zeros(n,1);

    if minORmax == "max"
        fun = @(x) -vec'*x;
    elseif minORmax == "min"
        fun = @(x) vec'*x;
    end

    [opt,val] = fmincon(fun,x0,[],[],ones(1,n),0,[],[],@nonlcon,options);

    if minORmax == "max"
        val = -val;
    end

end

