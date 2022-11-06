function res = KLdiv(P,Q,K)

    % P: probability distribution
    % Q: "nominal" probability distribution
    % K: number of sample data

    res = 0;
    for k = 1:K
        res = res + P(k)*log(P(k)/Q(k));
    end

end

