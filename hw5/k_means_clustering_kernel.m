function [Y] = k_means_clustering_kernel(X,K,sigma)
% K_MEANS_CLUSTERING_KERNEL
% Returns predicted groups Y at convergance.
% Input dataset X, K number of clusters, sigma for rbf kernel.
    % Numbers of observations in X
    N = size(X,1);

    % Create random centroid
    mu = datasample(X, K, 'Replace', false);

    % Create matrix assignees
    z_nk = zeros(N, K);
    z_nk_old = ones(N, K);
    
    % Create array for prediction
    Y = randi(K,N,1);
    
    % Define rbf kernel
    rbf_kernel = @(x1,x2) exp(-norm(x1-x2)^2/(2*sigma^2));
    
    % Iterate until assignments do not change
    while ~isequal(z_nk_old, z_nk)
        % Assign Y
        Y_old = Y;

        % Create term 3 for algorithm
        sum_ml = zeros(K,1);

        % Iterate over groups
        for k=1:K
            % Select data belonging to group
            X_k = X(Y_old == k,:);

            % Get number of observations in data
            N_k = size(X_k,1);

            % Iterate over observations
            for m=1:N_k
                % Iterate over observations
                for l=1:N_k
                    % Assign term3 value using algorithm
                    sum_ml(k) = sum_ml(k)+rbf_kernel(X_k(m,:),X_k(l,:));
                end
            end
            % Calculate total term3
            sum_ml(k) = sum_ml(k)/N_k^2;
        end

        % Iterate over observations
        for n=1:N
            % Select observation
            x_n = X(n,:);

            % Prepare distance matrix
            distances = zeros(K,1);

            % Iterate over groups
            for k=1:K
                % Select data belonging to group
                X_k = X(Y_old == k,:);

                % Get number of observations in group
                N_k = size(X_k,1);

                % Prepare term2 for assignment
                sum_m = 0;

                % Iterate over observations in group
                for m=1:N_k
                    % Assign term2
                    sum_m = sum_m+rbf_kernel(x_n,X_k(m,:));
                end

                % Complete term2 assignment
                sum_m = 2/N_k*sum_m;

                % Calculate distance
                distance = rbf_kernel(x_n, x_n)-sum_m+sum_ml(k);

                % Assign distance
                distances(k) = distance;
            end

            % Get index of minimum distances
            [~,idx] = min(distances);

            % Assign to prediciton
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
    end
end