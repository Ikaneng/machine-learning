% Style Guide: https://sites.google.com/site/matlabstyleguidelines/

% Clear Command Window
clc

% Retain current plot when adding new plots
hold on

% Generate 1000 points from 2D multivariate normal distribution
mu = [1; 1];
sigma = [0.1 -0.05; -0.05 0.2];
rng default  % For reproducibility
points = transpose(mvnrnd(mu, sigma, 1000));

% Scatter plot randomly generated points with points lying outside 
% f(x, 3) = 0 showing in black while points inside shown in blue
% points inside shown in blue.
% Credit: https://se.mathworks.com/matlabcentral/newsreader/view_thread/323289
pointsValues = arrayfun(@(x) f(points(:,x), 3), 1:size(points, 2));
pointsInside = pointsValues <= 0;
pointsOutside = pointsValues > 0;
pointSize = 10;
scatter(points(1, pointsInside), points(2, pointsInside), pointSize, 'b', 'filled');
scatter(points(1, pointsOutside), points(2, pointsOutside), pointSize, 'k', 'filled');

% Calculate limits of plot using min and max values of x and y
xMin = min(points(1, :));
yMin = min(points(2, :));
xMax = max(points(1, :));
yMax = max(points(2, :));
limits = [xMin, xMax, yMin, yMax];

% Disable warnings for fcontour
warning('off','all')

% Show the level sets f(x, r) = 0 for r = 1, 2, 3
r1 = fcontour(@(x,y) f([x;y], 1), limits, 'LevelList', 0, 'LineColor', 'm', 'LineWidth', 1.5);
r2 = fcontour(@(x,y) f([x;y], 2), limits, 'LevelList', 0, 'LineColor', 'c', 'LineWidth', 1.5);
r3 = fcontour(@(x,y) f([x;y], 3), limits, 'LevelList', 0, 'LineColor', 'r', 'LineWidth', 1.5);

% Enable warnings
warning('on','all')

% Title of the plot showing how many points lie outside f(x, 3) = 0
title(strcat(num2str(nnz(pointsOutside)),' points outside f(x, 3) = 0'), 'FontSize', 14);
xlabel('x');
ylabel('y');
zlabel('f(x,r)');
legend([r1, r2, r3], 'f(x,1)=0', 'f(x,2)=0', 'f(x,3)=0');

% Stop retaining current plot when adding new plots
hold off

% Define function f(x, r) from hw0.pdf
function y = f(x, r)
    mu = [1; 1];
    sigma = [0.1 -0.05; -0.05 0.2];
    y = (transpose(x - mu) * inv(sigma) * (x - mu)) ./ 2 - r;
end