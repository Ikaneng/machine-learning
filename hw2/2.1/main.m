%% 2.1 a, b, c unit test
clc

% Simple dataset
% X = [2,2,1;
%      2,2,100;
%      2,2,1000;
%      2,2,10000];
% 
% y = [1;
%      1;
%      -1;
%      -1];
% 
% Xtest = [2, 2, 100];

data = load('../dataset2.mat');
X = data.x;
y = data.y;

% test data
Xtest = [1.2683, 1.1287, 1.2233];
ytest = 1;


% Calculate mu and sigma for labels +1 and -1
[mu1, std1] = sge(X(y == 1,:));
sigma1 = std1.^2 * eye(size(X, 2));
[mu2, std2] = sge(X(y == -1,:));
sigma2 = std2.^2 * eye(size(X, 2));

[P1, P2, YtestBayes] = sph_bayes(Xtest, X, y, mu1, mu2, sigma1, sigma2)
[YtestClassifier] = new_classifier(Xtest, mu1, mu2)

if YtestBayes == ytest
    display('Bayes success');
else
    display('Bayes fail');
end

if YtestClassifier == ytest
    display('New Classifier success');
else
    display('New Classifier fail');
end

%% 2.1 d)
clc;

% Load data
data = load('../dataset2.mat');
X = data.x;
y = data.y;

% Choose number of folds
folds = 5;

% Store size of rows in X
Xrows = size(X,1);

% Create indices vector of size Xrows and folds numbers 
indices = crossvalind('Kfold', Xrows,folds);

% Create counters for errors
bayesErrors = 0;
newErrors = 0;

% Iterate over folds 
for fold = 1:folds
    % Select training data from X and y
    XtrainData = X(indices ~= fold, :);
    YtrainData = y(indices ~= fold, :);
    
    % Select (remaining) test data from X and y
    XtestData = X(indices == fold, :);
    YtestData = y(indices == fold, :);
    
    % Calculate mu and sigma for traning data having labels +1 and -1
    [mu1, std1] = sge(XtrainData(YtrainData == 1,:));
    sigma1 = std1.^2 * eye(size(XtrainData, 2));
    [mu2, std2] = sge(XtrainData(YtrainData == -1,:));
    sigma2 = std2.^2 * eye(size(XtrainData, 2));
    
    % Iterate over XtestData
    for test = 1:size(XtestData, 1)
        Xtest = XtestData(test, :);
        
        % Calculate label of Xtest using sph_bayes() and new_classifier()
        [~, ~, Ytest1] = sph_bayes(Xtest, XtrainData, YtrainData, mu1, mu2, sigma1, sigma2);
        [Ytest2] = new_classifier(Xtest, mu1, mu2);
        
        % Check if labels are incorrect, increment counters in this case
        if Ytest1 ~= YtestData(test)
            bayesErrors = bayesErrors + 1;
        end
        if Ytest2 ~= YtestData(test)
            newErrors = newErrors + 1;
        end
    end
end

% Print errors
display(bayesErrors);
display(newErrors);