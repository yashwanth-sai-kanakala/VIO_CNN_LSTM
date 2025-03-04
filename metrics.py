import numpy as np

def compute_ate(gt_poses, pred_poses):
    ate = np.mean(np.linalg.norm(gt_poses - pred_poses, axis=1))
    return ate

def compute_rpe(gt_poses, pred_poses):
    relative_gt = np.diff(gt_poses, axis=0)
    relative_pred = np.diff(pred_poses, axis=0)
    rpe = np.mean(np.linalg.norm(relative_gt - relative_pred, axis=1))
    return rpe
