function [P1, P2, Ytest] = sph_bayes(Xtest, X, y, mu1, mu2, sigma1, sigma2)
% SPH_BAYES Computes the probability of a new test point Xtest coming from
% class 1 (P1) or class 2 (P2). Finally, assigns a label Ytest to the test
% point based on the probabilities P1 and P2.

    % Calculate likelihoods
    L1 = mvnpdf(Xtest, mu1, sigma1);
    L2 = mvnpdf(Xtest, mu2, sigma2);
    
    % Calculate probability of Xtest being +1 and -1
    P1 = L1 / (L1 + L2);
    P2 = L2 / (L1 + L2);
    
    % Return the label Ytest which has highest probability
    if P1 > P2
        Ytest = 1;
    else
        Ytest = -1;
    end
    
end