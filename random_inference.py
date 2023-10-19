import random
import csv
import sys

# make random labels
def generate_random_labels(num_labels, num_classes):
    return [random.randint(0, num_classes - 1) for _ in range(num_labels)]


def save_random_labels_as_csv(num_labels, num_classes, output_file):
    random_labels = generate_random_labels(num_labels, num_classes)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file) 
        for i, label in enumerate(random_labels):
            writer.writerow([i, label])

num_labels = 269  # Number of our test data
num_classes = 26   # Number of classes of our data

x_test_path = sys.argv[1]
y_rand_pred_save_path = sys.argv[2]

# 무작위 레이블 생성 및 저장
save_random_labels_as_csv(num_labels, num_classes, y_rand_pred_save_path)