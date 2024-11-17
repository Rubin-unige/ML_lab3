%% Task 2: Build a kNN classifier

% kNN Classifier Function
% varargin so optional test labels can be input as well
function [predicted_labels, error_rate] = task2_kNNclassifier(train_data, train_labels, test_data, k, varargin)
    
    % Validate number of arguments
    if nargin < 4
        error("Not enough arguments. should be at least 4");
    end
    
    % check size of train and test columns
    if size(train_data, 2) ~= size(test_data, 2)
        error("Number of columns must be same in test and train data.");
    end
    
    % Validate value of k
    num_train_samples = size(train_data, 1);
    if k <= 0 || k > num_train_samples
        error("k must be a positive integer and less than or equal to the number of training samples.");
    end
    
    % Initialize predictions
    num_test_samples = size(test_data, 1);
    predicted_labels = zeros(num_test_samples, 1);
    
    % Loop over each test sample to find its k nearest neighbors
    for i = 1:num_test_samples
        % Compute Euclidean distance to all training samples
        distances = sqrt(sum((train_data - test_data(i, :)) .^ 2, 2));
        
        % Sort distances and get indices of k nearest neighbors
        [~, sorted_indices] = sort(distances);
        nearest_neighbors = train_labels(sorted_indices(1:k));
        % disp(nearest_neighbors);
        
        % Assign the most frequent label among the neighbors
        predicted_labels(i) = mode(nearest_neighbors);
    end
    
    % Calculate error rate if true test labels are provided
    if nargin == 5
        test_labels = varargin{1};
        if length(test_labels) ~= num_test_samples
            error("Length of test_labels must match the number of test samples.");
        end
        num_errors = sum(predicted_labels ~= test_labels);
        error_rate = num_errors / num_test_samples;
    else
        error_rate = NaN; % No error rate calculated if test_labels not provided
    end
end
