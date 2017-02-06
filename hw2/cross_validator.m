function [ error1counter, error2counter ] = cross_validator(X, y, folds, classifier)
    % CROSS_VALIDATOR 
    % Returns number of errors of classification of class 1 and to, error1 and error2,
    % by running k-folds cross-validation using training data X, class vector Y,
    % and a classification function classifier.
    % Requires X to be a matrix with dimensions samples x features, with
    % samples of class 1 and 2 separated as the first and second half of rows.
    % Requires y to be a row vector with corresponding labels +1 and -1 for X.

    % Store size of rows in X
    Xrows = size(X,1);

    % Create indices vector of size Xrows and folds numbers 
    indices = crossvalind('Kfold', Xrows,folds);

    % Create counters for errors
    error1counter = 0;
    error2counter = 0;

    % Iterate over folds 
    for fold = 1:folds
        % Select training data
        XtrainData = X(indices ~= fold, :);
        YtrainData = y(indices ~= fold, :);

        % Select (remaining) test data
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

            % Calculate label of Xtest
            [Ytest1] = classifier(Xtest, mu1, mu2);

            % Check if labels are incorrect, increment counters in this case
            if Ytest1 ~= YtestData(test)
                if (Ytest1 == 1)
                    error1counter = error1counter + 1;
                else
                    error2counter = error2counter + 1;
                end
            end
        end
    end

    % Return errors
    error1 = error1counter;
    error2 = error2counter;
end