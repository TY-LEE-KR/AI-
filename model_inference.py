import csv
import sys
import os
import torch
import timm
from train import MLP
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        img = []
        img_names = os.listdir(image_folder)
        for img_name in img_names:
            img.append(image_folder + '/' + img_name)
        self.images = sorted(img)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        img = Image.open(image)

        if img.mode == 'RGB':
            pass
        else:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img


def save_resluts_as_csv(results_list, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        for i, label in enumerate(results_list):
            writer.writerow([i, label])

def main():

    model_path = sys.argv[1]
    x_test_path = sys.argv[2]
    y_pred_save_path = sys.argv[3]

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

    # model = MLP()
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(768, 26, bias=True)
    model.load_state_dict(torch.load(model_path))

    test_dataset = TestDataset(x_test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # evalutation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results += predicted.tolist()

    save_resluts_as_csv(results, y_pred_save_path)

if __name__ == '__main__':
    main()