clear all, close all, clc

%% a)
% Load dataset
dataset = load('../dataset1.mat');

% Get data
data = dataset.x;

% Calculate mu and sigma
[mu, sigma] = sge(data);

% Plot prior and posterior distibutions with alpha = 10, beta = 1
xAxisScale = 0:0.01:10;
figure1 = figure(1)
plot(xAxisScale, priorDistribution(xAxisScale, 1, 1))
hold on
plot(xAxisScale, posteriorDistribution(xAxisScale, 1, 1))
xlabel('variance = s')
ylabel('P(variance = s)')
title('Prior & Posterior distributions, alpha = 1, beta = 1')
legend('Prior','Posterior')


% Plot prior and posterior distibutions with alpha = 1, beta = 1
xAxisScale = 0:0.001:1;
figure2 = figure(2)
plot(xAxisScale,priorDistribution(xAxisScale,10,1))
hold on
plot(xAxisScale,posteriorDistribution(xAxisScale,10,1))
xlabel('variance = s')
ylabel('P(variance = s)')
title('Prior & Posterior distributions, alpha = 10, beta = 1')
legend('Prior','Posterior')

%% b)

% Calculate data matrix dimensions
rowDimension = size(data, 1);
columnDimension = size(data, 2);

% Calculate s
s = sum(sum((data-repmat(mu, rowDimension, 1)).^2)) / columnDimension;

% Calculate s for Model A
sModelA = map(rowDimension,s,1,1)

% Calculate s for Model B
sModelB = map(rowDimension,s,10,1)

%% c)
% Calculated variables in b)
sModelA = 0.3547;
sModelB = 0.3516;
rowDimension = size(data, 1);

% Caclulate sA and sB
sA = sum((data(:, 1) - mu(1)).^2 + (data(:, 2) - mu(2)).^2)/(2 * sModelA);
sB = sum((data(:, 1) - mu(1)).^2 + (data(:, 2) - mu(2)).^2)/(2 * sModelB);

% Calculate bayes factor
bayesFactor = (sModelB / sModelA)^(rowDimension) * exp(sB - sA)

 