import torch
from torch.utils.data import DataLoader
from vio_model import MonocularVIOModel
from euroc_dataset import EuRoCMultiDataset
from metrics import compute_ate, compute_rpe
from trainer import plot_trajectory
import numpy as np
import cv2
import time


def run_saved_model(model_path, dataset_dirs, seq_len=10, image_size=(224, 224), batch_size=4):
    # Load the saved model
    model = MonocularVIOModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the dataset
    dataset = EuRoCMultiDataset(root_dirs=dataset_dirs, seq_len=seq_len, image_size=image_size, camera='cam0')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Process frame by frame

    # Inference
    all_gt_poses = []
    all_pred_poses = []
    with torch.no_grad():
        for images, imu_data, poses in dataloader:
            start_time = time.time()
            images, imu_data, poses = images.float(), imu_data.float(), poses.float()
            outputs = model(images, imu_data)
            end_time = time.time()

            fps = 1 / (end_time - start_time)

            gt_positions = poses[:, :, :3].cpu().numpy()
            pred_positions = outputs[:, :, :3].cpu().numpy()

            all_gt_poses.append(gt_positions)
            all_pred_poses.append(pred_positions)

            # Convert image tensor to numpy
            img = images[0, -1].permute(1, 2, 0).cpu().numpy() * 255
            img = img.astype(np.uint8)

            # Convert to BGR if needed
            if img.shape[-1] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Resize frame for better visibility
            img = cv2.resize(img, (640, 480))

            # Overlay text with predicted and ground truth positions
            gt_text = f"GT: x={gt_positions[0, -1, 0]:.2f}, y={gt_positions[0, -1, 1]:.2f}, z={gt_positions[0, -1, 2]:.2f}"
            pred_text = f"Pred: x={pred_positions[0, -1, 0]:.2f}, y={pred_positions[0, -1, 1]:.2f}, z={pred_positions[0, -1, 2]:.2f}"
            fps_text = f"FPS: {fps:.2f}"

            cv2.putText(img, gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, pred_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("VIO Inference", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

    # Concatenate all sequences
    all_gt_poses = np.concatenate(all_gt_poses, axis=0).reshape(-1, 3)
    all_pred_poses = np.concatenate(all_pred_poses, axis=0).reshape(-1, 3)

    # Compute metrics
    ate = compute_ate(all_gt_poses, all_pred_poses)
    rpe = compute_rpe(all_gt_poses, all_pred_poses)
    print(f"Inference ATE: {ate:.4f}, Inference RPE: {rpe:.4f}")

    # Plot trajectories
    plot_trajectory(all_gt_poses, all_pred_poses, save_path="inference_trajectory.png")


# Example usage
if __name__ == "__main__":
    model_path = r"C:\Users\pid50f0\Downloads\Yashwanth\cnn_lstm\Cnn-lstm checkpoints\checkpoint_19.pth"
    dataset_dirs = [r"C:\Users\pid50f0\Downloads\Yashwanth\vio\v2_01_easy\mav0"]
    run_saved_model(model_path, dataset_dirs)