%% 1.1)a-b)
% Clear environment
clear all;
close all;
clc;

% Control random number generation
rng(1);

% Load data
data = load('hw5_p1a.mat');
X = data.X;

% Choose k for k-means clustering
K = 2;

% Choose number of iterations to store y
store_iteration = 2;

% Calculate y prediction at convergance and after 2 iterations
[Y,Y_stored] = k_means_clustering(X,K,store_iteration);

% Select X where prediction at convergance ~= after 2 iterations
X_difference = X(Y~=Y_stored,:);

% Plot groups and difference
hold on;
gscatter(X(:,1),X(:,2),Y,'rb','.+');
plot(X_difference(:,1), X_difference(:,2), 'ko');

% Save figure
print('11abfigure','-dpng')

%% 1.1)c-d)
% Clear environment
clear all;
close all;
clc;

% Control random number generation
rng(1);

% Load data
data = load('hw5_p1b.mat');
X = data.X;

% Choose variables
K = 2;
sigma = 0.2;

% Calculate Ys
Y1 = k_means_clustering_kernel(X, K, sigma);
Y2 = k_means_clustering(X,K,0);

% Plot Ys
figure
subplot(1,2,1);
gscatter(X(:,1),X(:,2),Y1,'rb','.+');
subplot(1,2,2);
gscatter(X(:,1),X(:,2),Y2,'rb','.+');

% Save figure
print('11cdfigure','-dpng')