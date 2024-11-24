%% Task 3: Test the kNN classifier

function task3_testClassifier(x_train, y_train, x_test, y_test, k_values)
    % Determine the number of classes 
    num_classes = length(unique(y_train));

    % Initialize results storage
    predict_results = zeros(num_classes, length(k_values)); 
    confusion_matrices = cell(num_classes, length(k_values)); 
    precision = zeros(num_classes, length(k_values));
    recall = zeros(num_classes, length(k_values));
    f1_score = zeros(num_classes, length(k_values));

    % Binary classification (one vs all)
    for digit = 1:num_classes
        y_train_binary = (y_train == digit);
        y_test_binary = (y_test == digit);   

        % Loop over each k value
        for i = 1:length(k_values)
            k = k_values(i);

            % Call the kNN classifier from Task 2
            [predicted_labels, ~] = task2_kNNclassifier(x_train, y_train_binary, x_test, k);

            % Calculate accuracy and store the result
            accuracy_percentage = calculate_accuracy(predicted_labels, y_test_binary);
            predict_results(digit, i) = accuracy_percentage;

            % Calculate confusion matrix and store it
            cm = calculate_confusion_matrix(predicted_labels, y_test_binary);
            confusion_matrices{digit, i} = cm;

            % Calculate Precision, Recall, and F1 Score
            [precision(digit, i), recall(digit, i), f1_score(digit, i)] = calculate_metrics(cm);
        end
    end

    % Save the results
    save_results(predict_results, k_values, num_classes, confusion_matrices);
    
    % Plot the accuracy for each class for different k values
    plot_accuracy_for_each_class(predict_results, k_values, num_classes);
    
    % Compute and save precision, recall, f1 summaries
    compute_and_save_summaries(k_values, precision, recall, f1_score);
end


% Calculate Accuracy
function accuracy_percentage = calculate_accuracy(predicted_labels, y_test_binary)
    accuracy = sum(predicted_labels == y_test_binary) / length(y_test_binary);
    accuracy_percentage = accuracy * 100; 
end


% Calculate Confusion Matrix
function cm = calculate_confusion_matrix(predicted_labels, y_test_binary)
    cm = zeros(2, 2);  % Confusion matrix for binary classification
    cm(1, 1) = sum((y_test_binary == 0) & (predicted_labels == 0));  % True Negatives
    cm(1, 2) = sum((y_test_binary == 0) & (predicted_labels == 1));  % False Positives
    cm(2, 1) = sum((y_test_binary == 1) & (predicted_labels == 0));  % False Negatives
    cm(2, 2) = sum((y_test_binary == 1) & (predicted_labels == 1));  % True Positives
end


% Calculate Precision, Recall, and F1 Score
function [precision_val, recall_val, f1_score_val] = calculate_metrics(cm)
    TP = cm(2, 2);
    FP = cm(1, 2);
    FN = cm(2, 1);
    TN = cm(1, 1);

    precision_val = (TP + FP > 0) * (TP / (TP + FP));
    recall_val = (TP + FN > 0) * (TP / (TP + FN));
    f1_score_val = (precision_val + recall_val > 0) * (2 * (precision_val * recall_val) / (precision_val + recall_val));
end


% Plot Accuracy for Each Class
function plot_accuracy_for_each_class(predict_results, k_values, num_classes)
    for digit = 1:num_classes
        figure;
        plot(k_values, predict_results(digit, :), '-o', 'LineWidth', 1.5, 'MarkerEdgeColor', 'r');
        title(['Accuracy for Class ' num2str(digit)]);
        xlabel('k Value');
        ylabel('Accuracy (%)');
        grid on;
        
        class_file = fullfile('results/task3_results/accuracy_kvalues_each_class', ['accuracy_class_' num2str(digit) '_k_values.png']);
        saveas(gcf, class_file);  % Save figure as PNG for each class
        close(gcf);
    end
end


% Save Results to CSV
function save_results(predict_results, k_values, num_classes, confusion_matrices)
    % Save accuracy results
    k_accuracy_table = array2table(predict_results, ...
        'VariableNames', arrayfun(@num2str, k_values, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false));
    writetable(k_accuracy_table, 'results/task3_results/accuracy_kvalues_each_class/accuracyinpercet_k_table.csv', 'WriteRowNames', true);
    
    % Save confusion matrices
    for digit = 1:num_classes
        for i = 1:length(k_values)
            cm = confusion_matrices{digit, i};
            cm_table = array2table(cm, 'VariableNames', {'Predicted 0', 'Predicted 1'}, 'RowNames', {'Actual 0', 'Actual 1'});
            filename = sprintf('results/task3_results/confusion_matrix_each_class/confusion_matrix_class%d_k%d.csv', digit, k_values(i));
            writetable(cm_table, filename, 'WriteRowNames', true);
        end
    end
end


% Compute and Save Precision, Recall, and F1 Summaries
function compute_and_save_summaries(k_values, precision, recall, f1_score)
    % Precision Summary
    avg_precision = mean(precision, 1);
    std_precision = std(precision, 0, 1);
    percentile_25_precision = prctile(precision, 25, 1);
    percentile_75_precision = prctile(precision, 75, 1);

    % Recall Summary
    avg_recall = mean(recall, 1);
    std_recall = std(recall, 0, 1);
    percentile_25_recall = prctile(recall, 25, 1);
    percentile_75_recall = prctile(recall, 75, 1);

    % F1 Score Summary
    avg_f1 = mean(f1_score, 1);
    std_f1 = std(f1_score, 0, 1);
    percentile_25_f1 = prctile(f1_score, 25, 1);
    percentile_75_f1 = prctile(f1_score, 75, 1);

% Create Tables with k_values as the first column
precision_table = table(k_values', avg_precision', std_precision', percentile_25_precision', percentile_75_precision', ...
    'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});

recall_table = table(k_values', avg_recall', std_recall', percentile_25_recall', percentile_75_recall', ...
    'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});

f1_table = table(k_values', avg_f1', std_f1', percentile_25_f1', percentile_75_f1', ...
    'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});

% Display Tables
disp('Precision Summary:');
disp(precision_table);
disp('Recall Summary:');
disp(recall_table);
disp('F1-Score Summary:');
disp(f1_table);

% Save the results to CSV
writetable(precision_table, 'results/task3_results/precision_summary.csv', 'WriteRowNames', false);
writetable(recall_table, 'results/task3_results/recall_summary.csv', 'WriteRowNames', false);
writetable(f1_table, 'results/task3_results/f1_score_summary.csv', 'WriteRowNames', false);

end
