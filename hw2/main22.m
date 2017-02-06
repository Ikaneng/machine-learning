%% 2.2)a)
clc;

% Load dataset
digits = load('digits.mat');

% Transform dataset to form X: 2200 x 256; y: 2200 x 1
classFive = digits.data(:, :, 5)';
classEight = digits.data(:, :, 8)';
X = [classFive; classEight];
y = [repmat(1,1100,1); repmat(-1,1100,1)];

% Choose number of folds
folds = 5;

% Perform k-fold cross validation
[error1, error2] = cross_validator(X, y, folds, @(Xtest, mu1, mu2) new_classifier(Xtest, mu1, mu2));

% Print errors
totalErrors = error1 + error2;
errorRatio = (error1 + error2) / size(X, 1);
display(error1);
display(error2);
display(totalErrors);
display(errorRatio);

%% 2.2) b)
clc;

% Load dataset
digits = load('digits.mat');

% Transform dataset using an alternative feature function
imageDimension = 16;
featureVectorSize = 32;
classFive = digits.data(:, :, 5)';
classEight = digits.data(:, :, 8)';
classRows = size(classFive, 1);
altClassFive = zeros(classRows, featureVectorSize);
altClassEight = zeros(classRows, featureVectorSize);

% Iterate over sample data in classes
for sample = 1:classRows
    % Reshape sample data
    yFive = reshape(digits.data(:, sample, 5), imageDimension, imageDimension);
    yEight = reshape(digits.data(:, sample, 8), imageDimension, imageDimension);
    
    % Iterate over sample dimension
    for dimension = 1:imageDimension
        % Scale each pixel value to range [0, 1] from original gray-scale (0 ? 255)
        z = yFive(dimension, :) / 255;
        % Compute variance of each row of the image.
        altClassFive(sample, dimension) = var(z);
        % Repeat for classes
        z = yEight(dimension, :) / 255;
        altClassEight(sample, dimension) = var(z);
        
        % Scale each pixel value to range [0, 1] from original gray-scale (0 ? 255)
        z = yFive(:, dimension) / 255;
        % Compute variance of each column of the image.
        altClassFive(sample, 16 + dimension) = var(z);
        % Repeat for classes
        z = yEight(:, dimension) / 255;
        altClassEight(sample, 16 + dimension) = var(z);
    end
end

X = [altClassFive; altClassEight];
y = [repmat(1,1100,1); repmat(-1,1100,1)];

% Choose number of folds
folds = 5;

% Perform k-fold cross validation
[error1, error2] = cross_validator(X, y, folds, @(Xtest, mu1, mu2) new_classifier(Xtest, mu1, mu2));

% Print errors
totalErrors = error1 + error2;
errorRatio = (error1 + error2) / size(X, 1);
display(error1);
display(error2);
display(totalErrors);
display(errorRatio);