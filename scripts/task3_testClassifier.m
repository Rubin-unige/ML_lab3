function task3_testClassifier(x_train, y_train, x_test, y_test, k_values)
    % Function to test the kNN classifier for all classes and various k values.

    % Determine the number of unique classes 
    num_classes = length(unique(y_train));  % this way, both MNIST and wine data can be tested

    % Initialize results storage
    results = zeros(num_classes, length(k_values));  % To store accuracy for each class and k

    % Loop over each class
    for digit = 1:num_classes
        % Create binary labels for this class (1 vs. not this class)
        y_train_binary = (y_train == digit);  % Binary labels for training set
        y_test_binary = (y_test == digit);    % Binary labels for test set

        % Loop over each k value
        for i = 1:length(k_values)
            k = k_values(i);

            % Call the kNN classifier from Task 2
            [predicted_labels, ~] = task2_kNNclassifier(x_train, y_train_binary, x_test, k);

            % Calculate accuracy
            accuracy = sum(predicted_labels == y_test_binary) / length(y_test_binary);
            results(digit, i) = accuracy;  % Store the accuracy for this class and k value
        end
    end

    % Display results as a table
    disp('Accuracy Results (Rows: Classes, Columns: k values)');
    disp(array2table(results, 'VariableNames', arrayfun(@num2str, k_values, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false)));

    % Plot results for each class
    figure;
    for digit = 1:num_classes
        subplot(1, num_classes, digit);
        plot(k_values, results(digit, :), '-o', 'LineWidth', 2);
        title(['Class ' num2str(digit)]);
        xlabel('k Value');
        ylabel('Accuracy');
        grid on;
    end
end
