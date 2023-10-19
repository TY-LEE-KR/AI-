import os
import shutil

# 상위 폴더 경로
source_folder = "./train_ori"

# 대상 폴더 경로
destination_folder = "./train"

# 하위 폴더 이름 패턴 (L3_1부터 L3_26까지)
folder_patterns = [f"L3_{i}" for i in range(1, 27)]

# 각 하위 폴더를 반복하면서 이미지 파일을 train 폴더로 이동
for folder_pattern in folder_patterns:
    folder_path = os.path.join(source_folder, folder_pattern)
    
    if not os.path.exists(folder_path):
        continue  # 해당 패턴의 폴더가 존재하지 않으면 다음 패턴으로 이동
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_folder, file)
                shutil.move(source_file, destination_file)

print("이미지 파일을 train 폴더로 이동했습니다.")
