import os.path

from euroc_dataset import EuRoCMultiDataset
from vio_model import MonocularVIOModel
from trainer import train_and_evaluate
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

path = r'C:\Users\pid50f0\Downloads\Yashwanth\vio'
sequences = ["v1_01_easy", "v1_03_difficult", "v1_02_medium", "v2_01_easy", "v2_02_medium", "v2_03_difficult"]
train_dirs = [f"{path}/V1_03_difficult/mav0",f"{path}/v1_01_easy/mav0",f"{path}/v2_02_medium/mav0",f"{path}/v2_03_difficult/mav0",f"{path}/MH_01_easy/mav0",f"{path}/MH_03_medium/mav0",f"{path}/MH_04_difficult/mav0"]
val_dirs = [f"{path}/v2_01_easy/mav0",f"{path}/V1_02_medium/mav0",f"{path}/MH_02_easy/mav0",f"{path}/MH_05_difficult/mav0"]

checkpoint_path = r'C:\Users\pid50f0\Downloads\Yashwanth\cnn_lstm\checkpoints\checkpoint_11.pth'

train_dataset = EuRoCMultiDataset(root_dirs=train_dirs, seq_len=15, image_size=(224, 224), camera='cam0')
val_dataset = EuRoCMultiDataset(root_dirs=val_dirs, seq_len=15, image_size=(224, 224), camera='cam0')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MonocularVIOModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

start_epoch = 0

if os.path.exists(checkpoint_path):
    print("loading latest checkpoint.........")
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]+1
    print(f"Resuming model training from epoch: {start_epoch}")

train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs=100)
