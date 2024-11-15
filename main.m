%% Main Assignment Script

addpath("scripts");

%% Task 1: Obtain a data set

% Obtain train and test dataset from large MNIST dataset
[x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST] = task1_obtainMNISTdata();

% Obtain train and test dataset from large Wine dataset
[x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine] = task1_obtainWinedata();

%% Task 2: Build a kNN classifier

% Calling kNN classifier using large MNIST dataset
[predicted_labels_MINST, error_rate_MNIST] = task2_kNNclassifier(x_train_MNIST, y_train_MNIST, ...
    x_test_MNIST, y_test_MNIST, k, varargin);

% Display the prediction and error rate
disp(predicted_labels_MINST);
disp(error_rate_MNIST);

% Calling kNN classifier using large Wine dataset
[predicted_labels_Wine, error_rate_Wine] = task2_kNNclassifier(train_data, train_labels, ...
    test_data, y_test_Wine, k, varargin);

% Display the prediciton and error rate
disp(predicted_labels_Wine);
disp(error_rate_Wine);