#!/usr/bin/env python3

import numpy as np
import json
import pickle
import gtsam
from gtsam import symbol_shorthand, noiseModel, Marginals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from gtsam.utils import plot
from scipy.spatial.transform import Rotation

#camera
C = symbol_shorthand.C
#object
O = symbol_shorthand.O
#sfm noise (mildly arbitrary)
sfm_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))
#pose est noise (also mildly arbitrary)
pose_noise = noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))

def add_sfm_factors(graph, init_est, dataset_path, scale):
    f = open(dataset_path + '/reconstruction.json', 'r')
    data = json.load(f)

    images = data[0]['shots']
    for img in images.items():
        #print('Adding sfm factor for ' + img[0])
        img_num = int(img[0].split('.')[0])

        pos = np.array(img[1]['translation'])/scale
        rot_aa = np.array(img[1]['rotation']) #axis-angle
        rot_gtsam = gtsam.Rot3((Rotation.from_rotvec(rot_aa)).as_dcm())
        pos = (Rotation.from_rotvec(rot_aa)).inv().as_dcm() @ pos
        pos[2] = -pos[2]
        trans_gtsam = gtsam.Point3(pos)
        pose_gtsam = gtsam.Pose3(rot_gtsam, trans_gtsam)
        factor = gtsam.PriorFactorPose3(C(img_num), pose_gtsam, sfm_noise)
        graph.push_back(factor)
        init_est.insert(C(img_num), pose_gtsam)

def add_sfm_factors_gt(graph, init_est, dataset_path):
    f = open(dataset_path + '/ep_data.pkl', 'rb')
    data = pickle.load(f)

    for ind in range(len(data['cam_pose'])):
        pose_mat = np.eye(4)
        quat = data['cam_pose'][ind][3:]
        quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
        pose_mat[:3, :3] = (Rotation.from_quat(quat) * Rotation.from_euler('xyz', [0, np.pi, 0])).as_dcm()
        pose_mat[:3, 3] = data['cam_pose'][ind][:3]

        pos = pose_mat[:3, 3]
        rot_gtsam = gtsam.Rot3(pose_mat[:3, :3])
        trans_gtsam = gtsam.Point3(pos)
        pose_gtsam = gtsam.Pose3(rot_gtsam, trans_gtsam)
        factor = gtsam.PriorFactorPose3(C(ind), pose_gtsam, sfm_noise)
        graph.push_back(factor)
        init_est.insert(C(ind), pose_gtsam)

def add_pose_factors(graph, init_est, dataset_path):
    f = open(dataset_path + '/obj_det_poses.pkl', 'rb')
    data = pickle.load(f)
    cnt = 0
    for pose in data.items():
        #print('Adding pose det factor for ' + pose[0])
        num = int(pose[0].split('.')[0])

        pos = gtsam.Point3(pose[1][:3])
        quat = pose[1][3:]

        #wxyz to xyzw
        quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
        rot = gtsam.Rot3((Rotation.from_quat(quat)).as_dcm())
        pose = gtsam.Pose3(rot, pos)
        factor = gtsam.BetweenFactorPose3(C(num), O(0), pose, pose_noise)
        print('new factor')
        graph.push_back(factor)
        cnt += 1

    init_est.insert(O(0), gtsam.Pose3()) #identity

def plot_camera(pose, linecollection):
    transform = pose

    frustum_size = 0.05
    frustum_local = np.array([
        [0, 0, 0, 1],
        [-frustum_size*1.5, -frustum_size, frustum_size*2, 1],
        [0, 0, 0, 1],
        [-frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [0, 0, 0, 1],
        [frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [0, 0, 0, 1],
        [frustum_size*1.5, -frustum_size, frustum_size*2, 1],
        [-frustum_size*1.5, -frustum_size, frustum_size*2, 1],
        [-frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [-frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [frustum_size*1.5, frustum_size, frustum_size*2, 1],
        [frustum_size*1.5, -frustum_size, frustum_size*2, 1],
        [frustum_size*1.5, -frustum_size, frustum_size*2, 1],
        [-frustum_size*1.5, -frustum_size, frustum_size*2, 1]
    ])

    frustum_global = transform @ frustum_local.T

    linecollection = np.vstack([linecollection, frustum_global.T[None, :, :3]])
    return linecollection

def plot_object(pose, linecollection):
    transform = pose

    frustum_size = 0.05
    frustum_local = np.array([
        [frustum_size, -frustum_size, -frustum_size*2, 1],
        [-frustum_size, -frustum_size, -frustum_size*2, 1],
        [-frustum_size, frustum_size, -frustum_size*2, 1],
        [-frustum_size, frustum_size, -frustum_size*2, 1],
        [frustum_size, frustum_size, -frustum_size*2, 1],
        [frustum_size, frustum_size, -frustum_size*2, 1],
        [frustum_size, -frustum_size, -frustum_size*2, 1],
        [frustum_size, -frustum_size, -frustum_size*2, 1],
        [-frustum_size, -frustum_size, -frustum_size*2, 1],
        [frustum_size, -frustum_size, frustum_size*2, 1],
        [-frustum_size, -frustum_size, frustum_size*2, 1],
        [-frustum_size, frustum_size, frustum_size*2, 1],
        [-frustum_size, frustum_size, frustum_size*2, 1],
        [frustum_size, frustum_size, frustum_size*2, 1],
        [frustum_size, frustum_size, frustum_size*2, 1],
        [frustum_size, -frustum_size, frustum_size*2, 1],
        [frustum_size, -frustum_size, frustum_size*2, 1],
        [-frustum_size, -frustum_size, frustum_size*2, 1]
    ])

    frustum_global = transform @ frustum_local.T

    linecollection = np.vstack([linecollection, frustum_global.T[:, :3]])
    return linecollection

def visualize(gt_mat, meas_mat, graph, res):
    marginals = Marginals(graph, res)

    keys = res.keys()
    linecollection = np.zeros((0, 16, 3))
    linecollection2 = np.zeros((0, 16, 3))
    linecollectionobj = np.zeros((0, 3))
    linecollectionobj2 = np.zeros((0, 3))

    # Plot points and covariance matrices
    for pose in gt_mat[1:,:]:
        linecollection = plot_camera(pose, linecollection)
    linecollectionobj = plot_object(gt_mat[0], linecollectionobj)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    lc = Line3DCollection(linecollection, color='r')
    ax.add_collection(lc)

    # Plot points and covariance matrices
    for pose in meas_mat[1:,:]:
        linecollection2 = plot_camera(pose, linecollection2)
    linecollectionobj2 = plot_object(meas_mat[0], linecollectionobj2)

    lc2 = Line3DCollection(linecollection2, color='g')
    ax.add_collection(lc2)

    lco = Line3DCollection(linecollectionobj[None, :], color='r')
    ax.add_collection(lco)
    lco2 = Line3DCollection(linecollectionobj2[None, :], color='g')
    ax.add_collection(lco2)
    plt.show()

def compute_errors(graph, res, dataset_path):
    f = open(dataset_path + '/ep_data.pkl', 'rb')
    data = pickle.load(f)
    keys = res.keys()
    meas_poses = {}
    for key in keys:
        if not key == O(0):
            pose = res.atPose3(key)
            meas_poses[key] = [pose.translation(), Rotation.from_dcm(pose.rotation().matrix())]
    obj_pose = res.atPose3(O(0))
    meas_obj_pose = [obj_pose.translation(), Rotation.from_dcm(obj_pose.rotation().matrix())]

    gt_mat = np.zeros([0, 4, 4])
    meas_mat = np.zeros([0, 4, 4])

    meas_obj_pose_mat = np.eye(4)
    meas_obj_pose_mat[:3, :3] = meas_obj_pose[1].as_dcm()
    meas_obj_pose_mat[:3, 3] = meas_obj_pose[0]
    meas_mat = np.concatenate([meas_mat, meas_obj_pose_mat[None, :]], axis=0)

    gt_obj_pose_mat = np.eye(4)
    quat = data['obj_world_pose'][3:]
    quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
    gt_obj_pose_mat[:3, :3] = (Rotation.from_quat(quat) * Rotation.from_euler('xyz', [0, np.pi, 0])).as_dcm()
    gt_obj_pose_mat[:3, 3] = data['obj_world_pose'][:3]
    gt_mat = np.concatenate([gt_mat, gt_obj_pose_mat[None, :]], axis=0)

    for ind in range(len(data['cam_pose'])):
        meas_pose = meas_poses[C(ind)]
        meas_pose_mat = np.eye(4)
        meas_pose_mat[:3, :3] = meas_pose[1].as_dcm()
        meas_pose_mat[:3, 3] = meas_pose[0]
        meas_mat = np.concatenate([meas_mat, meas_pose_mat[None, :]], axis=0)

        pose_mat = np.eye(4)
        quat = data['cam_pose'][ind][3:]
        quat[0], quat[1], quat[2], quat[3] = quat[1], quat[2], quat[3], quat[0]
        pose_mat[:3, :3] = (Rotation.from_quat(quat) * Rotation.from_euler('xyz', [0, np.pi, 0])).as_dcm()
        pose_mat[:3, 3] = data['cam_pose'][ind][:3]
        gt_mat = np.concatenate([gt_mat, pose_mat[None, :]], axis=0)

    #align centroids
    gt_mat[:, :3, 3] -= np.mean(gt_mat[:, :3, 3], axis=0)
    meas_mat[:, :3, 3] -= np.mean(meas_mat[:, :3, 3], axis=0)
    #Kabsch Algorithm
    H = gt_mat[:, :3, 3].T @ meas_mat[:, :3, 3]
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    Rfull = np.eye(4)
    Rfull[:3, :3] = R
    gt_mat = Rfull @ gt_mat

    return gt_mat, meas_mat

def num_errors(gt_mat, meas_mat):
    obj_gt = gt_mat[0]
    obj_meas = meas_mat[0]
    err_cum = 0
    ang_err_cum = 0
    for i in range(gt_mat.shape[0]-1):
        ind = i + 1
        rel_trans_gt = np.linalg.inv(gt_mat[ind]) @ obj_gt
        rel_trans_meas = np.linalg.inv(meas_mat[ind]) @ obj_meas

        err = np.linalg.inv(rel_trans_gt) @ rel_trans_meas
        err_cum += np.linalg.norm(err[:3, 3])
        r = np.linalg.norm(Rotation.from_dcm(err[:3, :3]).as_rotvec())
        ang_err_cum += r*180/np.pi

    err_cum /= gt_mat.shape[0]-1
    ang_err_cum /= gt_mat.shape[0]-1
    print('translation error (m): ' + str(err_cum))
    print('rotation error (deg): ' + str(ang_err_cum))

if __name__ == '__main__':
    best_res = None
    best_err = np.inf
    for scale in np.linspace(1, 20, 1):
        graph = gtsam.NonlinearFactorGraph()
        init_est = gtsam.Values()
        #add_sfm_factors(graph, init_est, '../data/sim/ep_2', scale)
        add_sfm_factors_gt(graph, init_est, '../data/sim/ep_2')
        add_pose_factors(graph, init_est, '../data/sim/ep_2')

        params = gtsam.GaussNewtonParams()
        params.setRelativeErrorTol(1e-5)
        params.setMaxIterations(100)
        opt = gtsam.GaussNewtonOptimizer(graph, init_est, params)
        res = opt.optimize()

        if graph.error(res) < best_err:
            best_err = graph.error(res)
            best_res = res
            print(scale)
            print(best_err)

    gt_mat, meas_mat = compute_errors(graph, best_res, '../data/sim/ep_2')
    num_errors(gt_mat, meas_mat)
    visualize(gt_mat, meas_mat, graph, best_res)
