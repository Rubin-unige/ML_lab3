%% Task 1: Obtain a data set

% Obtain small wine dataset
function [x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine] = task1_obtainWinedata()

    % path to where wine data is located
    addpath("provided_scripts_data\small_wine_data\")

    % Load the wine dataset
    wine_data = load('wine.data');
    
    % Extract features and labels
    x_wine_data = wine_data(:, 2:end);  % Features 
    y_wine_data = wine_data(:, 1);      % Class
    
    % Normalise as suggested in remark
    x_norm = (x_wine_data - min(x_wine_data)) ./ (max(x_wine_data) - min(x_wine_data));
    
    n = length(y_wine_data);  
    
     % Set random seed to shuffle the training and test data
    rng(1);   
    indices = randperm(n);  
    
    % Since not specified, the data is split into 80% and 20% for now
    train_size = round(0.8 * n);  
    train_indices = indices(1:train_size);   
    test_indices = indices(train_size+1:end);

    % Training dataset
    x_train_Wine = x_norm(train_indices, :);  
    y_train_Wine = y_wine_data(train_indices);          
    
    % Test dataset
    x_test_Wine = x_norm(test_indices, :);    
    y_test_Wine = y_wine_data(test_indices);             
    
end