function [c,ceq] = nonlcon(x)

    c = [];
    ceq = norm(x) - 0.1;

end

