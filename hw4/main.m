%% 1.1
clc;
clear all;
close all;

xPositive = [2,2;4,4;4,0]
xNegative = [0,0;2,0;0,2]
yPositive = ones(size(xPositive,1),1)
yNegative = -1 * ones(size(xNegative,1),1)
X = [xPositive; xNegative]
Y = [yPositive; yNegative]

Mdl = fitcsvm(X,Y)
b = Mdl.Bias
w = Mdl.Beta
% Source: http://stackoverflow.com/questions/28556266/plot-svm-margins-using-matlab-and-libsvm
margin = margin(Mdl, X, Y) / norm(w)