%% 1.1)
clc;
clear all;
close all;

% Define points
xPositive = [2,2;4,4;4,0];
xNegative = [0,0;2,0;0,2];
yPositive = ones(size(xPositive,1),1);
yNegative = -1*ones(size(xNegative,1),1);
X = [xPositive; xNegative];
Y = [yPositive; yNegative];

% Calculate bias, beta, and margin
Mdl = fitcsvm(X,Y);
b = Mdl.Bias;
w = Mdl.Beta;
m = margin(Mdl,X,Y)/norm(w);

% Plot
hold on;
scatter(xPositive(:,1),xPositive(:,2),'r+');
scatter(xNegative(:,1),xNegative(:,2),'b.');
fplot(@(x1) -(w(1,1)*x1+b)/w(2,1));
plot([xPositive(1,1),(-b)/2],[xPositive(1,2),(-b)/2],'m:');
axis([(min(X(:,1))-1) (max(X(:,1))+1) (min(X(:,2))-1) (max(X(:,2))+1)]);
legend('1','-1','Boundary');
xlabel('x1');
ylabel('x2');
title(sprintf('Hyperplane: %d*x1+%d*x2+(%d)=0',w(1,1),w(2,1),b));
hold off;

% Save figure
print('11figure','-dpng')

%% 1.2)a)
% See report

%% 1.2)b)
clc;
clear all;
close all;

% Define points
xPositive = [2,2;4,4;4,0];
xNegative = [0,0;2,0;0,2];
yPositive = ones(size(xPositive,1),1);
yNegative = -1*ones(size(xNegative,1),1);
X = [xPositive;xNegative];
Y = [yPositive;yNegative];
 
% Find a minimum for problem
l = size(X,1);
H = eye(3);
f = zeros(3,1);
A = -diag(Y)*[X,ones(l,1)];
b = -ones(l,1);
min = quadprog(H,f,A,b)

%% 1.2)c)
% See report

%% 1.2)d)
clc;
clear all;
close all;

% Define points
xPositive = [2,2;4,4;4,0];
xNegative = [0,0;2,0;0,2];
yPositive = ones(size(xPositive,1),1);
yNegative = -1*ones(size(xNegative,1),1);
X = [xPositive;xNegative];
Y = [yPositive;yNegative];

% Find a minimum for problem
H = (Y*Y').*(X*X');
f = -ones(1,length(H));
A = zeros(1,length(H));
c = 0;
Aeq = Y';
ceq = 0;
lb = zeros(length(H),1);
ub = inf*ones(length(H),1);
alpha = quadprog(H,f,A,c,Aeq,ceq,lb,ub);
w = X'*(alpha.*Y)
b = 1-(X(1,:)*w)

%% 2.1)a-d)
clc;
clear all;
close all;

% Load data
data = load('d1b.mat');
X = data.X;
Y = data.Y;
boxconstraint = 1;

% Retain current plot when adding new plots
hold on

% Train binary support vector machine classifier
Mdl = fitcsvm(X,Y,'boxconstraint', boxconstraint);

% Calculate bias and margin
b  =  Mdl.Bias;
margin  =  2/norm(Mdl.Beta);

% Scatter plot by group
gscatter(X(:,1),X(:,2),Y,'rb','.+',10);

% Plot support vectors
plot(Mdl.SupportVectors(:,1),Mdl.SupportVectors(:,2),'go');

% Predict labels using support vector machine classification model
label = predict(Mdl,X);

% Create array of misclassified labels
mislabeled = Y-label;

% Plot misplaced labels
mislabeled_index  =  find(mislabeled); 
plot(X(mislabeled_index,1),X(mislabeled_index,2),'mo','MarkerSize',10);

% Plot hyperplane
fplot(@(x1) -(Mdl.Beta(1,1)*x1 + Mdl.Bias)/Mdl.Beta(2,1));

% Add plot info
legend('-1','1','Support Vectors','Mislabeled','Boundary');
xlabel('x1');
ylabel('x2');
title(sprintf('Boxconstraint = %d\nBias = %g\nMargin = %g',boxconstraint,b,margin));

% Set the hold state to off
hold off

% Save figure
print('21figure','-dpng')

%% 2.2)a)
clc;
clear all;
close all;

% Load data
data = load('d2.mat');
X = data.X;
Y = data.Y;

% Choose boxconstraint
boxconstraint = 1;

% Retain current plot when adding new plots
hold on

% Scatter plot by group
gscatter(X(:,1),X(:,2),Y,'rb','.+',10);

% Train support vector machine classifier
Mdl = svmtrain(X,Y,'boxconstraint',boxconstraint,'autoscale',false);

% Classify using support vector machine (SVM)
label = svmclassify(Mdl,X);

% Create array of misclassified labels
mislabeled = Y-label;

% Plot misplaced labels
mislabeled_index  =  find(mislabeled); 
plot(X(mislabeled_index,1),X(mislabeled_index,2),'mo');

% Report numbers
w = Mdl.SupportVectors'*Mdl.Alpha;
b = Mdl.Bias;

% Plot hyperplane
fplot(@(x1) -(w(1,1)*x1 + b)/w(2,1));

% Add plot info
legend('-1','1', 'Mislabeled', 'Boundary');
xlabel('x1');
ylabel('x2')
axis([(min(X(:,1))-1) (max(X(:,1))+1) (min(X(:,2))-1) (max(X(:,2))+1)]);
title(sprintf('Boxconstraint = %d',boxconstraint));

% Save figure
print('22afigure','-dpng')

%% 2.2)b)
clc;
clear all;
close all;

% Load data
data = load('d2.mat');
X = data.X;
Y = data.Y;

% Setup variables
showPlot = false;
folds = 5;
kernels = {'linear','quadratic','rbf'};
methods = {'smo','qp'};

% Save Command Window text to file
diary('22bCommandWindow.txt');    
diary on;

% Run cross validation and report execution time and error rate
for i = 1:length(kernels)
    for j = 1:length(methods)
        % Print current kernel and method
        fprintf('Kernel: %s, Method: %s\n',kernels{i},methods{j})
        
        % Start timer
        tic;
        
        % Calculate error count
        error_count = cross_validator(X,Y,folds,@(XTest,XTrain,YTrain)svm_classifier(kernels{i},methods{j},XTest,XTrain,YTrain,showPlot));
        
        % Stop and report timer
        toc;
        
        % Calculate error rate
        error_rate = error_count./size(Y,1);
        
        % Print error rate
        fprintf('Error rate: %g\n\n',error_rate);
    end
end

diary off;

%% 2.2)c)
% Use a grid approach, divide your 2D space into a k by k grid, evaluate ...
% that point as a sample in your SVM (i.e., predict the label), and plot ...
% the predictions at all points.
clc;
clear all;
close all;

% Load data
data = load('d2.mat');
X = data.X;
Y = data.Y;

% Retain current plot when adding new plots
hold on

% Scatter plot by group
gscatter(data.X(:,1),data.X(:,2),data.Y,'rb','.+',10);

% Set grid step size
gridStep = 0.1;

% Create meshgrid from X
[x1grid,x2grid] = meshgrid(min(data.X(:,1)):gridStep:max(data.X(:,1)), ...
    min(data.X(:,2)):gridStep:max(data.X(:,2)));
Xgrid = [x1grid(:),x2grid(:)]; 

% Train binary support vector machine classifier
Mdl = fitcsvm(X,Y,'KernelFunction','rbf');

% Predict scores using support vector machine classification model
[label,score] = predict(Mdl,Xgrid);       

% Contour plot of matrix
contour(x1grid,x2grid,reshape(score(:,2),size(x1grid)),[0 0],'k');

% Add plot info
title('RBF Kernel Decision Boundary');
legend('-1','1','Boundary');
xlabel('x1');
ylabel('x2');

% Set the hold state to off
hold off

% Save figure
print('22cfigure','-dpng')
