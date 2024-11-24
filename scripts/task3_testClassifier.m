%% Task 3: Test the kNN classifiers

% TODO: confusion matrix and statistics
function task3_testClassifier(x_train, y_train, x_test, y_test, initial_k)
    % Function to test the kNN classifier for all classes and various k values.

    % Determine the number of unique classes 
    num_classes = length(unique(y_train));  % This way, both MNIST and wine data can be tested

    % Initialize results storage
    initial_results = zeros(num_classes, length(initial_k));  % To store accuracy for each class and k

    % First pass: Loop over the initial k values
    for digit = 1:num_classes
        y_train_binary = (y_train == digit);  % Binary labels for training set
        y_test_binary = (y_test == digit);    % Binary labels for test set

        % Loop over each k value in the initial range
        for i = 1:length(initial_k)
            k = initial_k(i);

            % Call the kNN classifier from Task 2
            [predicted_labels, ~] = task2_kNNclassifier(x_train, y_train_binary, x_test, k);

            % Calculate accuracy
            accuracy = sum(predicted_labels == y_test_binary) / length(y_test_binary);
            accuracy_percentage = accuracy * 100;
            initial_results(digit, i) = accuracy_percentage;  % Store the accuracy for this class and k value
        end
    end

    % Save the initial results as a CSV
    initial_results_table = array2table(initial_results, ...
        'VariableNames', arrayfun(@num2str, initial_k, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false));
    writetable(initial_results_table, 'result/initial_k_results.csv', 'WriteRowNames', true);
    disp(initial_results_table);
    % Plot results for each class for the initial k values
    figure;
    for digit = 1:num_classes
        subplot(1, num_classes, digit);
        plot(initial_k, initial_results(digit, :), '-o', 'LineWidth', 2);
        title(['Class ' num2str(digit)]);
        xlabel('k Value');
        ylabel('Accuracy');
        grid on;
    end
    
    % Save the figure
    initial_k_file = fullfile('result', 'initial_k_results_plot.png'); 
    saveas(gcf, initial_k_file); % Save figure as PNG

    % Save the figure
    refined_k_file = fullfile('result', 'refined_k_results_plot.png'); 
    saveas(gcf, refined_k_file); % Save figure as PNG
    % Calculate the overall best k by averaging accuracies across all classes
    average_accuracy = mean(initial_results, 1);  % Average accuracy for each k value across all classes
    [best_accuracy, best_idx] = max(average_accuracy);  % Get the overall best k and corresponding accuracy
    best_k = initial_k(best_idx);  % Best k based on overall average accuracy

    % Display the overall best k and its accuracy
    fprintf('Overall Best k = %d with average accuracy = %.2f%%\n', best_k, best_accuracy);

end
