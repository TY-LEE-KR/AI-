import os
import csv

# 디렉토리 경로 설정
image_folder = "./dataset/test"
output_csv_file = "./dataset/y_test.csv"

# 라벨을 저장할 딕셔너리 초기화
label_dict = {}

# 이미지 폴더 내의 파일 목록 가져오기
image_files = os.listdir(image_folder)

# 파일 이름에서 라벨 추출 및 딕셔너리에 저장
for image_file in image_files:
    if image_file.startswith("i_") and image_file.endswith((".jpg", ".jpeg", ".png")):
        label = int(image_file.split("_")[1])
        label_dict[image_file] = label

# CSV 파일로 라벨 저장
with open(output_csv_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    for image_file, label in label_dict.items():
        csv_writer.writerow([image_file, label])

print(f"라벨을 {output_csv_file}에 저장했습니다.")