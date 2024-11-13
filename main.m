%% Main Assignment Script

addpath("scripts");

%% Task 1: Obtain a data set

% Obtain train and test dataset from large MNIST dataset
[x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST] = task1_obtainMNISTdata();

% Obtain train and test dataset from large Wine dataset
[x_train_Wine, y_train_Wine, x_test_Wine, y_test_Wine] = task1_obtainWinedata();

% Test if the data has been splited properly
disp(['Train MNIST size: ', num2str(size(x_train_MNIST, 1)), ' x ', num2str(size(x_train_MNIST, 2))]);
disp(['Test MNIST size: ', num2str(size(x_test_MNIST, 1)), ' x ', num2str(size(x_test_MNIST, 2))]);

disp(['Train Wine size: ', num2str(size(x_train_Wine, 1)), ' x ', num2str(size(x_train_Wine, 2))]);
disp(['Test Wine size: ', num2str(size(x_test_Wine, 1)), ' x ', num2str(size(x_test_Wine, 2))]);

%% Task 2: Build a kNN classifier


