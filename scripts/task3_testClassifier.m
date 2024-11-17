function task3_testClassifier(x_train, y_train, x_test, y_test, initial_k_values)
    % Function to test the kNN classifier for all classes and various k values.
    % Initial wide search to find the best overall k, then refine the search based on results.

    % Determine the number of unique classes 
    num_classes = length(unique(y_train));  % This way, both MNIST and wine data can be tested

    % Initialize results storage
    results = zeros(num_classes, length(initial_k_values));  % To store accuracy for each class and k

    % First pass: Loop over the initial k values
    for digit = 1:num_classes
        % Create binary labels for this class (1 vs. not this class)
        y_train_binary = (y_train == digit);  % Binary labels for training set
        y_test_binary = (y_test == digit);    % Binary labels for test set

        % Loop over each k value in the initial range
        for i = 1:length(initial_k_values)
            k = initial_k_values(i);

             % Check if k is divisible by num_classes
            if mod(k, num_classes) == 0
                fprintf('Skipping k = %d.\n', k);
                continue;  
            end

            % Call the kNN classifier from Task 2
            [predicted_labels, ~] = task2_kNNclassifier(x_train, y_train_binary, x_test, k);

            % Calculate accuracy
            accuracy = sum(predicted_labels == y_test_binary) / length(y_test_binary);
            accuracy_percentage = accuracy * 100;
            results(digit, i) = accuracy_percentage;  % Store the accuracy for this class and k value
        end
    end

    % Display results as a table for initial test
    disp('Initial Accuracy Results in percentage (Rows: Classes, Columns: k values)');
    disp(array2table(results, 'VariableNames', arrayfun(@num2str, initial_k_values, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false)));

    % Calculate the overall best k by averaging accuracies across all classes
    average_accuracy = mean(results, 1);  % Average accuracy for each k value across all classes
    [best_accuracy, best_idx] = max(average_accuracy);  % Get the overall best k and corresponding accuracy
    best_k = initial_k_values(best_idx);  % Best k based on overall average accuracy

    % Display the overall best k and its accuracy
    fprintf('Overall Best k = %d with average accuracy = %.2f%%\n', best_k, best_accuracy);

    % Now refine the k-values around the best k (±5 range)
    refined_k_values = (best_k - 5):(best_k + 5);  % Create range from best_k - 5 to best_k + 5
    refined_k_values = refined_k_values(refined_k_values > 0);  % Ensure k is positive
    refined_k_values = unique(refined_k_values);  % Remove duplicates, if any

    % Initialize a new results matrix for the refined k-values
    refined_results = zeros(num_classes, length(refined_k_values));  % To store accuracy for refined k values

    % Second pass: Loop over the refined k values
    for digit = 1:num_classes
        % Create binary labels for this class (1 vs. not this class)
        y_train_binary = (y_train == digit);  % Binary labels for training set
        y_test_binary = (y_test == digit);    % Binary labels for test set

        % Loop over each refined k value
        for i = 1:length(refined_k_values)
            k = refined_k_values(i);

            % Check if k is divisible by num_classes
            if mod(k, num_classes) == 0
                fprintf('Skipping k = %d.\n', k);
                continue;  
            end

            % Call the kNN classifier from Task 2
            [predicted_labels, ~] = task2_kNNclassifier(x_train, y_train_binary, x_test, k);

            % Calculate accuracy
            accuracy = sum(predicted_labels == y_test_binary) / length(y_test_binary);
            accuracy_percentage = accuracy * 100;
            refined_results(digit, i) = accuracy_percentage;  % Store the accuracy for this class and refined k value
        end
    end

    % Display refined results as a table
    disp('Refined Accuracy Results in percentage (Rows: Classes, Columns: refined k values)');
    disp(array2table(refined_results, 'VariableNames', arrayfun(@num2str, refined_k_values, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false)));

    % Plot results for each class for the refined k values
    figure;
    for digit = 1:num_classes
        subplot(1, num_classes, digit);
        plot(refined_k_values, refined_results(digit, :), '-o', 'LineWidth', 2);
        title(['Class ' num2str(digit)]);
        xlabel('k Value');
        ylabel('Accuracy');
        grid on;
    end
end
