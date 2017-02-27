function [ error_count ] = cross_validator(X, Y, folds, classifier)
    % CROSS_VALIDATOR 
    % Returns number of errors of classification of class, error_count,
    % by running k-folds cross-validation using training data X, class vector Y,
    % and a classification function classifier.
    % Requires X to be a matrix with dimensions samples x features.
    % Requires y to be a row vector with corresponding labels +1 and -1 for X.

    % Store size of rows in X
    Xrows = size(X,1);

    % Create indices vector of size Xrows and folds numbers 
    indices = crossvalind('Kfold', Xrows,folds);

    % Create counters for errors
    error_counter = 0;

    % Iterate over folds 
    for fold = 1:folds
        % Select training data
        XtrainData = X(indices ~= fold, :);
        YtrainData = Y(indices ~= fold, :);

        % Select (remaining) test data
        XtestData = X(indices == fold, :);
        YtestData = Y(indices == fold, :);

        % Iterate over XtestData
        for test = 1:size(XtestData, 1)
            Xtest = XtestData(test, :);

            % Calculate label of Xtest
            Ytest = classifier(Xtest, XtrainData, YtrainData);

            % Check if labels are incorrect, increment counters in this case
            if Ytest ~= YtestData(test)
                error_counter = error_counter + 1;
            end
        end
    end

    % Return errors
    error_count = error_counter;
end