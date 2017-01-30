% Inverse-gamma prior distibution
function [out] = priorDistribution(s, alpha, beta) 
    out = beta.^alpha/gamma(alpha)*s.^(-alpha-1).*exp(-beta./s);
end