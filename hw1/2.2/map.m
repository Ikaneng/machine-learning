% Maximum a posteriori probability (MAP) estimate
function [mapOutput] = map(rowDimension, sigma, alpha, beta)
    numerator = beta + sigma;
    denumerator = alpha + rowDimension + 1;
    mapOutput = numerator / denumerator;
end