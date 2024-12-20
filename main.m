%% Main Assignment Script

addpath("scripts");

% MNIST dataset implementation are commented
% uncomment them to use them
% MNIST dataset is too large, be prepared

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
% writetable(results_table_MNIST, 'results/task2_results/MNIST_classification_results.csv');
% % Save the error rate same file
% fileID = fopen('results/task2_results/MNIST_classification_results.csv', 'a');
% fprintf(fileID, '\nError Rate for MNIST dataset: %0.2f%%\n', errorRate_MNIST);
% fprintf('\nError Rate for MNIST dataset: %0.2f%%\n', errorRate_MNIST);
% fclose(fileID);

% Apply kNN classifier to Wine dataset
[predicted_Wine, errorRate_Wine] = task2_kNNclassifier(x_train_Wine, y_train_Wine, x_test_Wine, k, y_test_Wine);

% Display wine data results and save the results
disp("Predicted labels and Actual test labels for Wine dataset:");
% Create a table to display actual vs predicted labels
results_table_wine = table(y_test_Wine, predicted_Wine, 'VariableNames', {'Actual_Label', 'Predicted_Label'});
writetable(results_table_wine, 'results/task2_results/wine_classification_results.csv');
% Save the error rate same file
fileID = fopen('results/task2_results/wine_classification_results.csv', 'a');
fprintf(fileID, '\nError Rate for Wine dataset: %0.2f%%\n', errorRate_Wine);
fprintf('\nError Rate for Wine dataset: %0.2f%%\n', errorRate_Wine);
fclose(fileID);

%% Task 3: Test the kNN Classifier

% List of k values to test
k_values = [1:1:10, 11:2:21, 30, 40, 50];

% Call Task 3 function to test the kNN classifier on MNIST data
%task3_testClassifier(x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST, k_values);

% Call Task 3 function to test the kNN classifier on wine data
task3_testClassifier(x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine, k_values);
