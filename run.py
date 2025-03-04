import torch
from torch.utils.data import DataLoader
from vio_model import MonocularVIOModel
from euroc_dataset import EuRoCMultiDataset
from metrics import compute_ate, compute_rpe
from trainer import plot_trajectory
import numpy as np
import time

def run_saved_model(model_path, dataset_dirs,set, seq_len=10, image_size=(224, 224), batch_size=4):
    # Load the saved model
    model = MonocularVIOModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the dataset
    dataset = EuRoCMultiDataset(root_dirs=dataset_dirs, seq_len=seq_len, image_size=image_size, camera='cam0')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Inference
    all_gt_poses = []
    all_pred_poses = []
    start_time = time.time()
    with torch.no_grad():
        for images, imu_data, poses in dataloader:
            images, imu_data, poses = images.float(), imu_data.float(), poses.float()
            outputs = model(images, imu_data)

            # Extract ground truth and predicted positions (x, y, z)
            gt_positions = poses[:, :, :3].cpu().numpy()
            pred_positions = outputs[:, :, :3].cpu().numpy()

            all_gt_poses.append(gt_positions)
            all_pred_poses.append(pred_positions)
    end_time = time.time()
    inference_time = end_time - start_time

    # Concatenate all sequences
    all_gt_poses = np.concatenate(all_gt_poses, axis=0).reshape(-1, 3)
    all_pred_poses = np.concatenate(all_pred_poses, axis=0).reshape(-1, 3)

    # Compute metrics
    ate = compute_ate(all_gt_poses, all_pred_poses)
    rpe = compute_rpe(all_gt_poses, all_pred_poses)
    print(f"Inference ATE: {ate:.4f}, Inference RPE: {rpe:.4f}")
    print(f"Total Inference Time: {inference_time:.4f} seconds")

    # Plot trajectories
    plot_trajectory(all_gt_poses, all_pred_poses, save_path=f"vit_trajectory_{set}_ATE_{ate:.4f}_RPE_{rpe:.4f}_{inference_time:.4f}s.png")

# Example usage
if __name__ == "__main__":
    model_path = r"C:\Users\pid50f0\Downloads\Yashwanth\cnn_lstm\checkpoints\checkpoint_47.pth"
    for i in ['MH_01_easy','MH_03_medium','MH_05_difficult','MH_02_easy','MH_05_difficult',"v1_01_easy", "v1_03_difficult", "v1_02_medium", "v2_01_easy", "v2_02_medium", "v2_03_difficult"]:
        dataset_dirs = [rf"C:\Users\pid50f0\Downloads\Yashwanth\vio\{i}\mav0"]  # Specify validation dataset directories
        run_saved_model(model_path, dataset_dirs,i)
