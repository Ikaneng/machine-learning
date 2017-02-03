function [Ytest] = new_classifier(Xtest, mu1, mu2)
% NEW_CLASSIFIER assigns a label Ytest to the test
% point Xtest based on means from data sets with 2 different labels.
%   Classifier function
    b = (mu1 + mu2) ./ 2;
    Ytest = sign((mu1 - mu2) * transpose(Xtest - b) ./ norm((mu1 - mu2)));
end
