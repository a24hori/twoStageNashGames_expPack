function res_obj = RecourseValue(j,y,xi,K,Prob)

    res_obj = zeros(K,1);
    for k = 1:K
        res_obj(k) = Prob.gamma(j,y,xi,k);
    end

end

