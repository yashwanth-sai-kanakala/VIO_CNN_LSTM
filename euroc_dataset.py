import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2

class EuRoCMultiDataset(Dataset):
    def __init__(self, root_dirs, seq_len=10, image_size=(224, 224), camera='cam0'):
        self.root_dirs = root_dirs
        self.seq_len = seq_len
        self.image_size = image_size
        self.camera = camera

        self.image_timestamps, self.image_paths, self.imu_timestamps, self.imu_data, self.gt_timestamps, self.gt_data = self._load_all_data()

    def _load_all_data(self):
        all_image_timestamps = []
        all_image_paths = []
        all_imu_timestamps = []
        all_imu_data = []
        all_gt_timestamps = []
        all_gt_data = []

        for root_dir in self.root_dirs:
            image_timestamps, image_paths = self._load_image_paths(root_dir)
            imu_timestamps, imu_data = self._load_imu_data(root_dir)
            gt_timestamps, gt_data = self._load_ground_truth(root_dir)

            all_image_timestamps.extend(image_timestamps)
            all_image_paths.extend(image_paths)
            all_imu_timestamps.extend(imu_timestamps)
            all_imu_data.extend(imu_data)
            all_gt_timestamps.extend(gt_timestamps)
            all_gt_data.extend(gt_data)

        return (
            np.array(all_image_timestamps),
            all_image_paths,
            np.array(all_imu_timestamps),
            np.array(all_imu_data),
            np.array(all_gt_timestamps),
            np.array(all_gt_data),
        )

    def _load_image_paths(self, root_dir):
        """Load image file paths and timestamps from the camera folder."""
        image_dir = os.path.join(root_dir, self.camera, "data")
        timestamps_path = os.path.join(root_dir, self.camera, "data.csv")

        # Skip the header row
        timestamps = pd.read_csv(timestamps_path, header=None, skiprows=1)[0].astype(np.int64).values

        # Get image paths
        image_paths = [os.path.join(image_dir, f"{ts}.png") for ts in timestamps]

        return timestamps, image_paths

    def _load_imu_data(self, root_dir):
        """Load IMU data and timestamps."""
        imu_path = os.path.join(root_dir, "imu0", "data.csv")
        
        # Skip the header row
        imu_data = pd.read_csv(imu_path, header=None, skiprows=1).values

        # Columns: timestamp (ns), accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
        timestamps = imu_data[:, 0].astype(np.int64)  # Ensure timestamps are integers
        imu_data = imu_data[:, 1:]  # Remove timestamp column

        return timestamps, imu_data

    def _load_ground_truth(self, root_dir):
        """Load ground truth pose data and timestamps."""
        gt_path = os.path.join(root_dir, "state_groundtruth_estimate0", "data.csv")
        
        # Skip the header row
        gt_data = pd.read_csv(gt_path, header=None, skiprows=1).values

        # Columns: timestamp (ns), p_x, p_y, p_z, q_x, q_y, q_z, q_w, v_x, v_y, v_z, accel_bias, gyro_bias
        timestamps = gt_data[:, 0].astype(np.int64)  # Ensure timestamps are integers
        gt_data = gt_data[:, 1:8]  # Extract position (p_x, p_y, p_z) and orientation (q_x, q_y, q_z, q_w)

        return timestamps, gt_data



    def _synchronize_data(self, target_timestamps, source_timestamps, source_data):
        synchronized_data = []
        for ts in target_timestamps:
            idx = np.argmin(np.abs(source_timestamps - ts))
            synchronized_data.append(source_data[idx])
        return np.array(synchronized_data)

    def __len__(self):
        return len(self.image_paths) - self.seq_len

    def __getitem__(self, idx):
        image_seq = []
        for i in range(idx, idx + self.seq_len):
            img = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.image_size)
            img = img.astype(np.float32) / 255.0  # Normalize pixel values

            # Convert single channel to 3 channels
            img = np.stack([img, img, img], axis=-1)  # Shape: (H, W, 3)
            image_seq.append(img)

        image_seq = np.stack(image_seq, axis=0).transpose(0, 3, 1, 2)  # Shape: (seq_len, 3, H, W)
        target_timestamps = self.image_timestamps[idx:idx + self.seq_len]
        imu_seq = self._synchronize_data(target_timestamps, self.imu_timestamps, self.imu_data)
        pose_seq = self._synchronize_data(target_timestamps, self.gt_timestamps, self.gt_data)

        return (
            torch.tensor(image_seq),
            torch.tensor(imu_seq),
            torch.tensor(pose_seq)
        )
