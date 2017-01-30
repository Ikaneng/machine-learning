% Plot data
function [] = plot_data(data)


% Retain current plot when adding new plots
hold on

% Calculate mu and sigma
[mu, sigma] = sge(data);

% Centers of circles
centers = [mu; mu; mu];

% Radii of circles
radiuses = [sigma; sigma; sigma];

% Factor for circles
k = [1; 2; 3];

% Function returning the distance from the centre for a point
distanceFromCentre = @(a,b)(a-mu(1)).^2 + (b-mu(2)).^2;

% Array of distances from centre for points in dataset
distancesFromCentre = arrayfun(@(x) distanceFromCentre(data(x,1), data(x,2)), 1:size(data,1));

% Iterate using different k values
for k=1:size(k)
    % Get points inside circle
    pointsInside = sum(distancesFromCentre<=(k*sigma)^2);
    
    % Get points outside circle
    pointsOutside = sum(distancesFromCentre>(k*sigma)^2);
    
    % Get fraction of points outside circle
    fraction = pointsOutside/(pointsOutside + pointsInside);
    
    % Array of points with 0.01 increments from 0 to 2pi
    theta = 0:0.01:2*pi;
    
    % Plot line of concentric circle
    h(k)= plot(mu(1)+(k*sigma)*cos(theta), mu(2)+(k*sigma)*sin(theta));
    % Legend info for each circle
    functionInfo{k} = [strcat('Data outside circle: ', num2str(fraction))];
end

% Draw legend and scatter plot
legend([h(1),h(2),h(3)],functionInfo);
scatter(data(:,1), data(:,2));

 
hold off
end