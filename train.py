import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler
import csv
import sys
from PIL import Image
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="Palette images with Transparency expressed in bytes should be converted to RGBA images")

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        label = []
        img = []
        f = open(label_file, 'r')
        rdr = csv.reader(f)
        for line in rdr:
            img.append(image_folder + '/' + line[0])
            label.append(line[1])
        self.images = img
        self.labels = label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx]) - 1

        img = Image.open(image)

        if img.mode == 'RGB':
            pass
        else:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 26)

    def forward(self, x):
        x = x.view(-1, 64 * 64 * 3)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def main():
    
    image_folder = sys.argv[1]
    label_file = sys.argv[2]
    model_save_path = sys.argv[3]

    train_dataset = CustomDataset(image_folder, label_file, transform=transform)
    sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)

    # model = MLP()
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(768, 26, bias=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for n,p in model.named_parameters():
        if "head" in n:
            p.requires_grad=True
        else:
            p.requires_grad=False
    nn.init.xavier_uniform(model.head.weight)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\n{epoch+1} epoch start")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")
    
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()