%% Main Assignment Script

addpath("scripts");

%% Task 1: Obtain a data set

% Obtain train and test dataset from large MNIST dataset
[x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST] = task1_obtainMNISTdata();

% Obtain train and test dataset from large Wine dataset
[x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine] = task1_obtainWinedata();

%% Task 2: Build a kNN classifier

% Parameter
k = 3; % initial value of k, increase oddly like {5, 7 ,..}

% % Apply kNN classifier to MNIST dataset
% [predicted_MNIST, errorRate_MNIST] = task2_kNNclassifier(x_train_MNIST, y_train_MNIST, x_test_MNIST, k, y_test_MNIST);
% 
% % Display results
% disp('Error Rate for MNIST in percentage:');
% disp(errorRate_MNIST);

% Apply kNN classifier to Wine dataset
[predicted_Wine, errorRate_Wine] = task2_kNNclassifier(x_train_Wine, y_train_Wine, x_test_Wine, k, y_test_Wine);

% Display results
disp('Error Rate for Wine in percentage:');
disp(errorRate_Wine);
 
%% Task 3: Test the kNN Classifier

% List of k values to test
k_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50];

% Call Task 3 function to test the kNN classifier on MNIST data
% task3_testClassifier(x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST, k_values);

% Call Task 3 function to test the kNN classifier on Wine data
task3_testClassifier(x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine, k_values);