%% 1.1)
clc;
clear all;
close all;

% Define points
xPositive=[2,2;4,4;4,0];
xNegative=[0,0;2,0;0,2];
yPositive=ones(size(xPositive,1),1);
yNegative=-1*ones(size(xNegative,1),1);
X=[xPositive; xNegative];
Y=[yPositive; yNegative];

% Calculate bias, beta, and margin
Mdl=fitcsvm(X,Y);
b=Mdl.Bias;
w=Mdl.Beta;
m=margin(Mdl,X,Y)/norm(w);

% Plot
hold on;
scatter(xPositive(:,1),xPositive(:,2),'b+');
scatter(xNegative(:,1),xNegative(:,2),'r.');
syms x1 x2;
ezplot(w(1,1)*x1+w(2,1)*x2+b==0);
plot([xPositive(1,1), (-b)/2],[xPositive(1,2), (-b)/2],'m:');
axis([(min(X(:,1))-1) (max(X(:,1))+1) (min(X(:,2))-1) (max(X(:,2))+1)]);
hold off;

%% 1.2)a)
% See report

%% 1.2)b)
clc;
clear all;
close all;

% Define points
xPositive=[2,2;4,4;4,0];
xNegative=[0,0;2,0;0,2];
yPositive=ones(size(xPositive,1),1);
yNegative=-1*ones(size(xNegative,1),1);
X=[xPositive;xNegative];
Y=[yPositive;yNegative];
 
% Find a minimum for problem
l=size(X,1);
H=eye(3);
f=zeros(3,1);
A=-diag(Y)*[X,ones(l,1)];
b=-ones(l,1);
min=quadprog(H,f,A,b)

%% 1.2)c)
% See report

%% 1.2)d)
clc;
clear all;
close all;

% Define points
xPositive=[2,2;4,4;4,0];
xNegative=[0,0;2,0;0,2];
yPositive=ones(size(xPositive,1),1);
yNegative=-1*ones(size(xNegative,1),1);
X=[xPositive;xNegative];
Y=[yPositive;yNegative];

% Find a minimum for problem
H=(Y*Y').*(X*X');
f=-ones(1,length(H));
A=zeros(1,length(H));
c=0;
Aeq=Y';
ceq=0;
lb=zeros(length(H),1);
ub=inf*ones(length(H),1);
alpha=quadprog(H,f,A,c,Aeq,ceq,lb,ub);
w=X'*(alpha.*Y)
b=1-(X(1,:)*w)

%% 2.1)a-d)
clc;
clear all;
close all;

% Load data
data=load('d1b.mat');
X=data.X;
Y=data.Y;

% Retain current plot when adding new plots
hold on

% Train binary support vector machine classifier
Mdl=fitcsvm(X,Y);

% Scatter plot by group
gscatter(X(:,1),X(:,2),Y,'rb','.+',6);

% 2-D line plot
plot(Mdl.SupportVectors(:,1),Mdl.SupportVectors(:,2),'o');

% Predict labels using support vector machine classification model
label=predict(Mdl,X);

% Create array of misclassified labels
mislabeled=Y-label;

% Plot misplaced labels
mislabeled_index = find(mislabeled); 
plot(X(mislabeled_index,1),X(mislabeled_index,2),'.r','MarkerSize',21);

% Create symbolic variables and functions 
syms x1 x2;

% Plots hyperplane
ezplot(Mdl.Beta(1,1) * x1 + Mdl.Beta(2,1) * x2 + Mdl.Bias == 0);

% Add plot info
legend('-1','1','Support vectors', 'Mislabeled','Boundary');
xlabel('x1');
ylabel('x2');
title('Box-constraint = 1');

% Set the hold state to off
hold off

% Report numbers
boxconstraints = Mdl.BoxConstraints
bias = Mdl.Bias
margin = 2 / norm(Mdl.Beta)

%% 2.2)a)
clc;
clear all;
close all;

% Load data
data=load('d2.mat');
X=data.X;
Y=data.Y;

% Retain current plot when adding new plots
hold on

% Train support vector machine classifier
Mdl=svmtrain(X, Y);

% Scatter plot by group
gscatter(X(:,1),X(:,2),Y,'rb','.+',6);

% Classify using support vector machine (SVM)
label=svmclassify(Mdl,X,'ShowPlot',true);

% Create array of misclassified labels
mislabeled=Y-label;

% Plot misplaced labels
mislabeled_index = find(mislabeled); 

% Report numbers
w=Mdl.SupportVectors'*Mdl.Alpha;
b=Mdl.Bias;

plot(X(mislabeled_index,1),X(mislabeled_index,2),'o');

% Add plot info
legend('-1','1', 'Mislabeled');
xlabel('x1');
ylabel('x2')
title('Box-constraint = 1');

%% 2.2)b)
clc;
clear all;
close all;

% Load data
data=load('d2.mat');
X=data.X;
Y=data.Y;

% Setup variables
showPlot=false;
folds=5;
kernels={'linear','quadratic','rbf'};
methods={'smo','qp'};

% Run cross validation and report execution time and error rate
for i=1:length(kernels)
    for j=1:length(methods)
        % Print current kernel and method
        fprintf('Kernel: %s, Method: %s\n',kernels{i},methods{j})
        
        % Start timer
        tic;
        
        % Calculate error count
        error_count=cross_validator(X,Y,folds,@(XTest,XTrain,YTrain)svm_classifier(kernels{i},methods{j},XTest,XTrain,YTrain,showPlot));
        
        % Stop and report timer
        toc;
        
        % Calculate error rate
        error_rate=error_count./size(Y,1);
        
        % Print error rate
        fprintf('Error rate: %g\n\n',error_rate);
    end
end

%% 2.2)c)
clc;
clear all;
close all;

% Load data
data=load('d2.mat');
X=data.X;
Y=data.Y;

% Retain current plot when adding new plots
hold on

% Scatter plot by group
gscatter(data.X(:,1),data.X(:,2),data.Y,'rb','.+',6);

% Set grid step size
s=0.02;

% Create meshgrid from X
[x1grid,x2grid] = meshgrid(min(data.X(:,1)):s:max(data.X(:,1)), ...
    min(data.X(:,2)):s:max(data.X(:,2)));
Xgrid = [x1grid(:),x2grid(:)];        

% Train binary support vector machine classifier
Mdl=fitcsvm(data.X,data.Y,'KernelFunction','rbf');

% Predict scores using support vector machine classification model
[~,score]=predict(Mdl,Xgrid);

% Contour plot of matrix
contour(x1grid,x2grid,reshape(score(:,2),size(x1grid)),[0 0],'k');

% Add plot info
title('Decision Boundary for RBF kernel')
legend('-1','1','Boundary');
xlabel('x1');
ylabel('x2')

% Set the hold state to off
hold off
