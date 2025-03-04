import torch
from torch.utils.data import DataLoader
from euroc_dataset import EuRoCMultiDataset
from vio_model import MonocularVIOModel
from metrics import compute_ate, compute_rpe
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

image_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.GaussianBlur(kernel_size=3)
])

# Apply transformation to each image in the dataset
def apply_transforms(images):
    return torch.stack([image_transforms(img) for img in images])

def augment_imu(imu_data, noise_std=0.01):
    noise = torch.randn_like(imu_data) * noise_std
    return imu_data + noise


def plot_trajectory(gt_poses, pred_poses, save_path=None):
    """
    Plot ground truth and predicted trajectories.
    
    Args:
        gt_poses (np.ndarray): Ground truth positions (N, 3).
        pred_poses (np.ndarray): Predicted positions (N, 3).
        save_path (str): Path to save the plot (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(gt_poses[:, 0], gt_poses[:, 1], label="Ground Truth", color="blue")
    plt.plot(pred_poses[:, 0], pred_poses[:, 1], label="Predicted", color="red", linestyle="dashed")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    # plt.close()


from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, epochs,  save_path=None):
    patience = 28
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_ate = float('inf')  # Track best ATE
    no_improvement_epochs = 0  # Track epochs with no ATE improvement

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        for images, imu_data, poses in train_loader:
            images = apply_transforms(images).to(device).float()
            imu_data = augment_imu(imu_data).to(device).float()
            poses = poses.to(device).float()

            optimizer.zero_grad()
            outputs = model(images, imu_data)
            loss = criterion(outputs, poses)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.6f}")

        # Validation phase
        model.eval()
        all_gt_poses = []
        all_pred_poses = []
        with torch.no_grad():
            for images, imu_data, poses in val_loader:
                images, imu_data, poses = images.to(device).float(), imu_data.to(device).float(), poses.to(
                    device).float()
                outputs = model(images, imu_data)

                # Extract ground truth and predicted positions (x, y, z)
                gt_positions = poses[:, :, :3].cpu().numpy()
                pred_positions = outputs[:, :, :3].cpu().numpy()

                all_gt_poses.append(gt_positions)
                all_pred_poses.append(pred_positions)

        # Concatenate all sequences
        all_gt_poses = np.concatenate(all_gt_poses, axis=0).reshape(-1, 3)
        all_pred_poses = np.concatenate(all_pred_poses, axis=0).reshape(-1, 3)

        # Compute metrics
        ate = compute_ate(all_gt_poses, all_pred_poses)
        rpe = compute_rpe(all_gt_poses, all_pred_poses)
        print(f"Validation ATE: {ate:.4f}, Validation RPE: {rpe:.4f}")

        # Adjust learning rate based on validation loss
        scheduler.step(ate)
        plot_trajectory(all_gt_poses, all_pred_poses, save_path=f"eval_plots/trajectory_epoch_{epoch + 1}.png")
        # Save the best model
        if ate < best_ate:
            best_ate = ate
            no_improvement_epochs = 0  # Reset counter
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ate": best_ate,
            }
            torch.save(ckpt, f"checkpoints/checkpoint_{epoch + 1}.pth")
            print(f"Model saved with ATE: {ate:.4f}")
        else:
            no_improvement_epochs += 1

        # Early stopping
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs of no improvement.")
            break