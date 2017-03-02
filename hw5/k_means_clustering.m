function [Y, Y_stored] = k_means_clustering(X,K,store_iteration)
% K_MEANS_CLUSTERING
% Returns predicted groups Y at convergance, predicted groups Y_stored at
% iteration store_iteration.
% Input dataset X, K number of clusters, store_iteration for which
% iteration to store predicted groups Y_stored.
    % Numbers of observations in X
    N = size(X,1);
    
    % Create arrays for prediction
    Y = zeros(N, 1);
    Y_stored = zeros(N, 1);

    % Create random centroid
    mu = datasample(X, K, 'Replace', false);

    % Create matrix assignees
    z_nk = zeros(N, K);
    z_nk_old = ones(N, K);
    
    % Create iterator counter
    iterator = 0;
    
    % Iterate until assignments do not change
    while ~isequal(z_nk_old, z_nk)
        
        % Increment iterator
        iterator = iterator + 1;
        
        % Assign centroid
        mu_old = mu;
        
        % Iterate over observations
        for n=1:N
            % Select observation in X
            x = X(n,:);

            % Create array for distances in groups
            distances = zeros(K, 1);

            % Iterate over groups
            for i=1:K
                % Calculate centeroid of group
                mu_k = mu_old(i, :);

                % Calculaate distance of point to centeroid
                distance = norm(x - mu_k);

                % Update distance for point
                distances(i) = distance;
            end

            % Find indices for minimum distances
            [~, idx] = min(distances);

            % Set index in Y prediction
            Y(n) = idx;
        end
        
        % Iterate over groups
        for i=1:K
            % Select points of group from data
            x_k = X(Y == i,:);

            % Update group centeroid to mean of these points
            mu(i,:) = mean(x_k);
        end
        
        % Assign matrix assignee
        z_nk_old = z_nk;
        
        % Restore matrix assigner
        z_nk = zeros(N, K);
        
        % Iterate over observations
        for i = 1:N
            % Mark assigner observations according to predicted groups
            z_nk(i,Y(i)) = 1;
        end

        % Check if on store_iteration
        if iterator == store_iteration
            % Store prediction
            Y_stored = Y;
        end
    end
end