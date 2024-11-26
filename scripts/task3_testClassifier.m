%% Task 3: Test the kNN classifier

function task3_testClassifier(x_train, y_train, x_test, y_test, k_values)
    % Determine the number of classes 
    num_classes = length(unique(y_train));

    % Initialize results storage
    confusion_matrices = cell(num_classes, length(k_values)); 
    accuracy = zeros(num_classes, length(k_values)); 
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

            % Create table with binary classification comparing binary test and predicted test
            % this will help create confusion matrix and evaluate them easily
            results_table = table(y_test_binary, predicted_labels, 'VariableNames', {'Actual_Label', 'Predicted_Label'});
            writetable(results_table, sprintf(['results/task3_results/predicted_values_each_k/' ...
                'wine_classification_results_k%d_class%d.csv'], k, digit));

            % Calculate confusion matrix and store it
            cm = calculate_confusion_matrix(predicted_labels, y_test_binary);
            confusion_matrices{digit, i} = cm;

            % Calculate Accuracy, Precision, Recall, and F1 Score
            [accuracy(digit, i), precision(digit, i), recall(digit, i), f1_score(digit, i)] = calculate_metrics(cm);
        end
    end
    
    % Save the confusion matrix
    save_confusion_matrix(k_values, num_classes, confusion_matrices);

    % for each k and each class we visualise the value of accuracy, precision, recall and F1 score
    % Accuracy
    plot_metric_for_each_class(accuracy, k_values, num_classes, 'Accuracy', 'results/task3_results/accuracy_kvalues_each_class');   
    save_and_display_metric_table(accuracy, k_values, num_classes, ...
    'Accuracy', 'results/task3_results/accuracy_kvalues_each_class/accuracy_percent_k_table.csv');
    % Precision
    plot_metric_for_each_class(precision, k_values, num_classes, 'Precision', 'results/task3_results/precision_kvalues_each_class');
    save_and_display_metric_table(precision, k_values, num_classes, ...
        'Precision', 'results/task3_results/precision_kvalues_each_class/precision_percent_k_table.csv');
    % Recall
    plot_metric_for_each_class(recall, k_values, num_classes, 'Recall', 'results/task3_results/recall_kvalues_each_class');
    save_and_display_metric_table(recall, k_values, num_classes, ...
        'Recall', 'results/task3_results/recall_kvalues_each_class/recall_percent_k_table.csv');
    % F1 Score
    plot_metric_for_each_class(f1_score, k_values, num_classes, 'F1 Score', 'results/task3_results/f1_kvalues_each_class');
    save_and_display_metric_table(f1_score, k_values, num_classes, ...
        'F1 Score', 'results/task3_results/f1_kvalues_each_class/f1_score_percent_k_table.csv');

    % Compute and save average, standard deviation and percentiles for each k
    compute_and_save_summaries(k_values, accuracy, precision, recall, f1_score);
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
function [accuracy_val, precision_val, recall_val, f1_score_val] = calculate_metrics(cm)
    TP = cm(2, 2);
    FP = cm(1, 2);
    FN = cm(2, 1);
    TN = cm(1, 1);

    accuracy_val = (TP + TN) / (TP + FP + TN + FN);
    accuracy_val = accuracy_val * 100;  % Convert to percentage
    precision_val = (TP + FP > 0) * (TP / (TP + FP));
    recall_val = (TP + FN > 0) * (TP / (TP + FN));
    f1_score_val = (precision_val + recall_val > 0) * (2 * (precision_val * recall_val) / (precision_val + recall_val));
end

function plot_metric_for_each_class(metric_results, k_values, num_classes, metric_name, save_folder)
    for digit = 1:num_classes
        figure;
        plot(k_values, metric_results(digit, :), '-o', 'LineWidth', 1.5, 'MarkerEdgeColor', 'r');
        title([metric_name ' for Class ' num2str(digit)]);
        xlabel('k Value');
        ylabel(metric_name);
        grid on;

        % Save figure as PNG
        class_file = fullfile(save_folder, [lower(metric_name) '_class_' num2str(digit) '_k_values.png']);
        saveas(gcf, class_file);
        close(gcf);
    end
end

function save_and_display_metric_table(metric_results, k_values, num_classes, metric_name, save_path)
    % Create table for the metric
    metric_table = array2table(metric_results, ...
        'VariableNames', arrayfun(@num2str, k_values, 'UniformOutput', false), ...
        'RowNames', arrayfun(@num2str, 1:num_classes, 'UniformOutput', false));
    
    % Display table in command window
    disp(['Table for ' metric_name ':']);
    disp(metric_table);
    
    % Save the table to a CSV file
    writetable(metric_table, save_path, 'WriteRowNames', true);
end

% Save Results to CSV
function save_confusion_matrix(k_values, num_classes, confusion_matrices)
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
function compute_and_save_summaries(k_values, accuracy, precision, recall, f1_score)

    % Accuracy Summary
    avg_accuracy = mean(accuracy, 1);
    std_accuracy = std(accuracy, 0, 1);
    percentile_25_accuracy = prctile(accuracy, 25, 1);
    percentile_75_accuracy = prctile(accuracy, 75, 1);

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

    % Create Tables
    accuracy_table = table(k_values', avg_accuracy', std_accuracy', percentile_25_accuracy', percentile_75_accuracy', ...
        'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});

    precision_table = table(k_values', avg_precision', std_precision', percentile_25_precision', percentile_75_precision', ...
        'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});
    
    recall_table = table(k_values', avg_recall', std_recall', percentile_25_recall', percentile_75_recall', ...
        'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});
    
    f1_table = table(k_values', avg_f1', std_f1', percentile_25_f1', percentile_75_f1', ...
        'VariableNames', {'k Value', 'Average', 'StdDev', '25th Percentile', '75th Percentile'});
    
    % Display Tables
    disp('Accuracy Summary:');
    disp(accuracy_table);
    disp('Precision Summary:');
    disp(precision_table);
    disp('Recall Summary:');
    disp(recall_table);
    disp('F1-Score Summary:');
    disp(f1_table);
    
    % Save the results to CSV
    writetable(accuracy_table, 'results/task3_results/accuracy_summary.csv', 'WriteRowNames', false);
    writetable(precision_table, 'results/task3_results/precision_summary.csv', 'WriteRowNames', false);
    writetable(recall_table, 'results/task3_results/recall_summary.csv', 'WriteRowNames', false);
    writetable(f1_table, 'results/task3_results/f1_score_summary.csv', 'WriteRowNames', false);
end
