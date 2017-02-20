%% 2.1 
% See report

%% 2.1.1
clc;
clear all;

% Should output:
% cost on training/validation/test data is 2.302585
% classification error rate on training/validation/test data is 0.900000
net(0, 0, 0, 0, 0, false, 0);

%% 2.2
% Result: 
% The cost on the training data is 2.768381
clc;
clear all;

% Test with huge weight decay, should pass 
% net(1e7, 7, 10, 0, 0, false, 4);

% Test with no weight decay, should pass
% net(0, 7, 10, 0, 0, false, 4);

% Experiment test 1
% net(0, 10, 70, 0.005, 0, false, 4);

% Experiment test 2
% net(0, 10, 70, 0.5, 0, false, 4);

% Test with small weight decay, report results
net(0.1, 7, 10, 0, 0, false, 4);

%% 2.3 a-b)
% Results:
% a) Best with momentum
% b) Optimal with learning_rate = 0.2;
clc;
clear all;

% Choose net arguments
wd_coefficient = 0;
n_hid = 10;
n_iters = 70;
do_early_stopping = false;
mini_batch_size = 4;

% Create array of moments
moments = [0.0, 0.9];

% Create array of learning rates
learning_rates = [0.002, 0.01, 0.05, 0.2, 1.0, 5.0, 20.0];

% Loop over moments
for i = 1:length(moments)
    % Assign momentum
    momentum_multiplier = moments(i);
    
    % Loop over learning rates
    for j = 1:length(learning_rates)
        % Assign learning rate
        learning_rate = learning_rates(j)
        
        % Call net function
        net(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size);
    end
end

%% 2.4 a)
% Result:
% The cost on the validation data is 0.430185
clc;
clear all;

net(0, 200, 1000, 0.35, 0.9, false, 100);

%% 2.4 b)
% Result: 
% The cost on the validation data is 0.334505
clc;
clear all;

net(0, 200, 1000, 0.35, 0.9, true, 100);

%% 2.4 c)
% Result:
% wd_coefficient = 0.001
% The classification cost (i.e. without weight decay) on the validation data is 0.287910
clc;
clear all;

% Create array of weight decay coefficients
wd_coefficients = [10, 1, 0.0001, 0.001, 5];

% Loop over weight decay values
for i = 1:length(wd_coefficients)
    % Assign weight decay
    wd_coefficient = wd_coefficients(i)
    
    % Call net function
    net(wd_coefficient, 200, 1000, 0.35, 0.9, false, 100);
end

%% 2.4 d)
% Result:
% n_hid = 30
% The cost on the validation data is 0.317077
clc;
clear all;

% Create array of number of hidden units
n_hids = [10, 30, 100, 130, 200];

% Loop over hidden unit numbers
for i = 1:length(n_hids)
    % Assign number of hidden units
    n_hid = n_hids(i)
    
    % Call net function
    net(0, n_hid, 1000, 0.35, 0.9, false, 100);
end

%% 2.4 e)
% Result:
% n_hid = 37
% The cost on the validation data is 0.265165
clc;
clear all;

% Create array of number of hidden units
n_hids = [18, 37, 83, 113, 236];

% Loop over hidden unit numbers
for i = 1:length(n_hids)
    % Assign number of hidden units
    n_hid = n_hids(i)
    
    % Call net function
    net(0, n_hid, 1000, 0.35, 0.9, true, 100);
end

%% 2.4 f)
% Result:
% The classification error rate on the test data is 0.073222
clc;
clear all;

% Choose best working net arguments from previous questions
wd_coefficient = 0.001;
n_hid = 37;
n_iters = 1000;
learning_rate = 0.2;
momentum_multiplier = 0.9;
do_early_stopping = true;
mini_batch_size = 100;

% Call net function
net(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size);
