% Style Guide: https://sites.google.com/site/matlabstyleguidelines/

% Clear Command Window
clc

% Load variables from file into workspace
xData = load('dataset0.txt');

% xData Covariance
xCovariance = cov(xData)

% xData Correlation coefficients
xCorrelation = corrcoef(xData)

% Scale each feature in X between [0, 1] to obtain a new dataset Y
yData = normalizeColumns(xData)

% yData Covariance
yCovariance = cov(yData)

% yData Correlation coefficients
yCorrelation = corrcoef(yData)

% Plot Covariance & Correlation of xData & yData 
figure;

subplot(2,2,1)
imagesc(xCovariance)
title('xData Covariance')
colorbar

subplot(2,2,2)
imagesc(yCovariance)
title('yData Covariance')
colorbar

subplot(2,2,3)
imagesc(xCorrelation)
title('xData Correlation')
colorbar

subplot(2,2,4)
imagesc(yCorrelation)
title('yData Correlation')
colorbar

% Find pair of features in yData having minimum correlation by
% finding smallest element in yCorrelation
minCorrelation = min(min(yCorrelation));

% Find indices and values of minCorrelation elements, occurrs twice in
% correlation matrix
[feature1, feature2] = find(yCorrelation == minCorrelation)

% Plot scatterplot of features where these elements occurr
figure;
scatter (yData(:,feature1(1)), yData(:, feature2(1)), 10, 'red', 'filled');
title(sprintf('yData minimum correlation features: %i, %i; Correlation = %.4f', feature1(1), feature2(1), minCorrelation));

%% Functions
function normalizedColumns = normalizeColumns(m)
    normalizedMatrix = [];
    for i = 1 : size(m, 2)
        normalizedMatrix(:,i) = m(:, i) ./ max(m(:, i));
    end
    normalizedColumns = normalizedMatrix;
end