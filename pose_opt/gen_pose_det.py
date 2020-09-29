#!/usr/bin/env python3

import numpy as np
import pickle
from scipy.spatial.transform import Rotation

def load_poses(path):
    f = open(path + '/ep_data.pkl', 'rb')
    data = pickle.load(f)

    gt_mat = np.zeros([0, 4, 4])
    gt_obj_pose_mat = np.eye(4)
    quat = data['obj_world_pose'][3:]
    quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
    gt_obj_pose_mat[:3, :3] = (Rotation.from_quat(quat) * Rotation.from_euler('xyz', [0, np.pi, 0])).as_dcm()
    gt_obj_pose_mat[:3, 3] = data['obj_world_pose'][:3]
    gt_mat = np.concatenate([gt_mat, gt_obj_pose_mat[None, :]], axis=0)

    for ind in range(len(data['cam_pose'])):
        pose_mat = np.eye(4)
        quat = data['cam_pose'][ind][3:]
        quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
        pose_mat[:3, :3] = (Rotation.from_quat(quat) * Rotation.from_euler('xyz', [0, np.pi, 0])).as_dcm()
        pose_mat[:3, 3] = data['cam_pose'][ind][:3]
        gt_mat = np.concatenate([gt_mat, pose_mat[None, :]], axis=0)

    return gt_mat[0], gt_mat[1:]

def save_poses(obj_pose, cam_poses, path):
    f = open(path + '/obj_det_poses.pkl', 'wb')
    data = {}
    for ind, cam_pose in enumerate(cam_poses):
        diff = np.linalg.inv(cam_pose) @ obj_pose
        R = diff[:3, :3]
        t = diff[:3, 3]
        R = Rotation.from_euler('xyz', np.random.normal(0, 0.1, [3])).as_dcm() @ R
        t += np.random.normal(0, 0.05, [3])
        quat = Rotation.from_dcm(R).as_quat()
        quat[1], quat[2], quat[3], quat[0] = quat[0], quat[1], quat[2], quat[3]
        total_vec = np.hstack([t, quat])
        data[str(ind) + '.png'] = total_vec

    pickle.dump(data, f)

if __name__ == '__main__':
    obj_pose, cam_poses = load_poses('../data/sim/ep_2')
    save_poses(obj_pose, cam_poses, '../data/sim/ep_2')