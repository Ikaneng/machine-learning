%% 1.1
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
margin=margin(Mdl,X,Y)/norm(w);

% Plot
hold on;
scatter(xPositive(:,1),xPositive(:,2),'b+');
scatter(xNegative(:,1),xNegative(:,2),'r.');
syms x1 x2;
ezplot(1*x1+1*x2+b==0);
plot([xPositive(1,1), (-b)/2],[xPositive(1,2), (-b)/2],'m:');
axis([(min(X(:,1))-1) (max(X(:,1))+1) (min(X(:,2))-1) (max(X(:,2))+1)]);
hold off;

%% 1.1)a)
% See report

%% 1.1)b)
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

%% 1.1)c)
% See report

%% 1.1)d)
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
w=X'*(alpha.*Y);
b=1-(X(1,:)*w);