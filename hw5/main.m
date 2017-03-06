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
legend('1','-1',sprintf('Differance from iteration %d', store_iteration));
xlabel('x1');
ylabel('x2');
title('Linear k-means clustering');
hold off;

% Save figure
print('11abfigure','-dpng');

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
title('Kernel k-means clustering');
xlabel('x1');
ylabel('x2');
axis('square');
subplot(1,2,2);
gscatter(X(:,1),X(:,2),Y2,'rb','.+');
title('Linear k-means clustering');
xlabel('x1');
ylabel('x2');
axis('square');

% Save figure
print('11cdfigure','-dpng');

%% 1.2)a)
% Clear environment
clear all;
close all;
clc;

% Save Command Window text to file
diary('12aCommandWindow.txt');    
diary on;

% Control random number generation
rng(3);

% Load data
load('medium_100_10k.mat');

% Choose K for kmeans clustering
K = 10;

% Choose number of words
number_of_words = 10;

% Calculate distances
[~,~,~,D] = kmeans(wordembeddings,K);

% Create cell array for topWords
closest_words = cell(number_of_words,K);

% Iterate over clusters in distances
for cluster = 1:size(D,2)

    % Calculate sorting indices
    [~, sort_index] = sort(D(:,cluster),'ascend');

    % Get indices in number_of_words
    index = sort_index(1:number_of_words);

    % Print cluster number
    fprintf('Cluster %d \n', cluster);

    % Iterate over number of words
    for i = 1:number_of_words

        % Display word with index
        disp(vocab{index(i)});

        % Insert word into topwords
        closest_words{cluster,i} = vocab{index(i)};
    end

    % Print whitespace
    fprintf('\n');
end

diary off;

%% 1.2)b)
% Clear environment
clear all;
close all;
clc;

% Control random number generation
rng(1);

% Load data
load('medium_100_10k.mat');

% Choose K for kmeans clustering
K = 10;

% Choose number of runs
R = 10;

% Create matrix of fractions for each run
F = zeros(R,1);

% Iterate over number of runs
for r = 1:R
    % Find cluster 1
    C_1 = kmeans(wordembeddings,K,'Replicates',1);
    
    % Calculate N_0
    C_1_class = C_1(strcmp(vocab, 'cavalry'));
    C_1_indices = find(C_1 == C_1_class);
    C1_size = size(C_1_indices, 1);
    N_0 = (C1_size^2-C1_size)/2;

    % Find cluster 2
    [C_2] = kmeans(wordembeddings,K, 'Replicates', 1);
    
    N1 = 0;
    % Iterate over groups
    for k = 1:K
        % Calculate N_1
        temporary = find(C_2(C_1_indices) == k);
        N1 = N1+sum(1:length(temporary)-1);
    end
    
    % Add fraction to F
    F(r) = N1/N_0;
end

% Display the average F-value
disp(sum(F)/R);

%% 1.2)c)
% Clear environment
clear all;
close all;
clc;

% Control random number generation
rng(1);

% Load data
load('medium_100_10k.mat');

% Choose K for kmeans clustering
K = 10;

% Choose number of samples
S = 1000;

% Get random samples
random_samples = randsample(size(vocab,2),S);

% Classify samples
idx = kmeans(wordembeddings(random_samples,:),K,'Replicates',1);

% Project embeddings into 2D
projection = tsne(wordembeddings(random_samples,:));

% Scatter plot by group
gscatter(projection(:,1),projection(:,2),idx);
title('Visualization of word embeddings');
xlabel('x1');
ylabel('x2');

% Save figure
print('12cfigure','-dpng');