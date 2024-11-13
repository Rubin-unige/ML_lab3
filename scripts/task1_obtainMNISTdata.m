%% Task 1: Obtain a dataset

% Obtain large MNIST dataset
function [x_train_MNIST, y_train_MNIST, x_test_MNIST, y_test_MNIST] = task1_obtainMNISTdata()

    % path to where MNIST data is located
    addpath("provide_scripts_data\large_MINST_data\");

    % Load training dataset using loadMNIST function, 0 loads the training data
    [x_train_MNIST, y_train_MNIST] = loadMNIST(0); 

    % Load test dataset using loadMNIST function, 1 loads the test data
    [x_test_MNIST, y_test_MNIST] = loadMNIST(1);   

end
