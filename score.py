import csv
import sys

y_test_path = "./dataset/y_test.csv"
y_pred_path = "y_pred.csv"
score_save_path = "model_score.txt"
# y_test_path = sys.argv[1]
# y_pred_path = sys.argv[2]
# score_save_path = sys.argv[3]


def read_csv_file(file_path):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            data.append(row)
    return data

def calculate_accuracy(actual_labels, predicted_labels):
    correct_predictions = 0
    total_predictions = len(actual_labels)

    for actual, predicted in zip(actual_labels, predicted_labels):
        if (actual-1) == predicted:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions)
    return accuracy

test_y = read_csv_file(y_test_path)
anon_y = read_csv_file(y_pred_path)

test_y = [int(row[1]) for row in test_y]
anon_y = [int(row[1]) for row in anon_y]

# calculate Acc
accuracy = calculate_accuracy(test_y, anon_y)

with open(score_save_path, 'w') as file:
    file.write(str(accuracy))