%% Main Assignment Script

addpath("scripts");

%% Task 1: Obtain a data set

% Obtain train and test dataset from large MNIST dataset
[x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST] = task1_obtainMNISTdata();

% Obtain train and test dataset from large Wine dataset
[x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine] = task1_obtainWinedata();

%% Task 2: Build a kNN classifier

% Parameter
k = 3; % initial value of k

% % Apply kNN classifier to MNIST dataset
% [predicted_MNIST, errorRate_MNIST] = task2_kNNclassifier(x_train_MNIST, y_train_MNIST, x_test_MNIST, k, y_test_MNIST);

% % Display MNIST data results and save the results
% disp("Predicted labels and Actual test labels for MNIST dataset:");
% % Create a table to display actual vs predicted labels
% results_table_MNIST = table(y_test_MNIST, predicted_MNIST, 'VariableNames', {'Actual_Label', 'Predicted_Label'});
% % Write the table to a CSV file
% writetable(results_table_MNIST, 'results/task2_results/MNIST_classification_results.csv');
% % Save the error rate same file
% fileID = fopen('results/task2_results/MNIST_classification_results.csv', 'a');
% fprintf(fileID, '\nError Rate for MNIST dataset: %0.2f%%\n', errorRate_MNIST);
% fclose(fileID);

% Apply kNN classifier to Wine dataset
[predicted_Wine, errorRate_Wine] = task2_kNNclassifier(x_train_Wine, y_train_Wine, x_test_Wine, k, y_test_Wine);

% Display wine data results and save the results
disp("Predicted labels and Actual test labels for Wine dataset:");
% Create a table to display actual vs predicted labels
results_table_wine = table(y_test_Wine, predicted_Wine, 'VariableNames', {'Actual_Label', 'Predicted_Label'});
% Write the table to a CSV file
writetable(results_table_wine, 'results/task2_results/wine_classification_results.csv');
% Save the error rate same file
fileID = fopen('results/task2_results/wine_classification_results.csv', 'a');
fprintf(fileID, '\nError Rate for Wine dataset: %0.2f%%\n', errorRate_Wine);
fclose(fileID);

%% Task 3: Test the kNN Classifier

% List of k values to test
% k_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50];

% Call Task 3 function to test the kNN classifier on MNIST data
%task3_testClassifier(x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST, k_values);

% Call Task 3 function to test the kNN classifier on wine data
% task3_testClassifier(x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine, k_values);


% Perform Leave-One-Out Cross Validation (LOO-CV) for wine data
% n = length(y_train_Wine);  % Number of training samples
% for i = 1:n
%     % Leave out the i-th sample for testing
%     x_train_loo = x_train_Wine([1:i-1, i+1:end], :);
%     y_train_loo = y_train_Wine([1:i-1, i+1:end]);
%     x_test_loo = x_train_Wine(i, :);  % The i-th sample as test
%     y_test_loo = y_train_Wine(i);     % The corresponding label
% 
%     % Call task3_testClassifier with LOO data
%     task3_testClassifier(x_train_loo, y_train_loo, x_test_loo, y_test_loo, k_values);
% end