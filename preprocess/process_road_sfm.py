import argparse
import numpy as np
from read_write_model import *
import pyquaternion
import argparse
import os, time
from plyfile import PlyData, PlyElement
import open3d as o3d
import cv2
import tqdm

def project_point(X, q, t, K):
    R = pyquaternion.Quaternion(q).rotation_matrix
    x = K @ (R @ X + t)
    x = x[:2] / x[2]
    return int(round(x[0])), int(round(x[1]))

def remove_z_outliers(pcd, radius=0.2, z_thresh=0.05):
    """
    根据局部邻域的Z高度差剔除离群点
    :param pcd: open3d.geometry.PointCloud
    :param radius: 邻域搜索半径 (单位与点云坐标一致)
    :param z_thresh: 超过该Z差值的点认为是异常
    :return: 清理后的点云
    """
    pts = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    keep_mask = np.zeros(len(pts), dtype=bool)

    for i, p in enumerate(pts):
        [_, idx, _] = kdtree.search_radius_vector_3d(p, radius)
        if len(idx) < 3:
            continue
        z_neighbors = pts[idx, 2]
        z_med = np.median(z_neighbors)
        if abs(p[2] - z_med) < z_thresh:
            keep_mask[i] = True
    keep_idx = np.where(keep_mask)[0]
    return pcd.select_by_index(keep_idx), keep_idx

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automatically reorient colmap')
    
    # Add command-line argument(s)
    parser.add_argument('--project_dir', required=True, help="project directory")
    args = parser.parse_args()
    images_dir = os.path.join(args.project_dir, "camera_calibration/rectified/images")
    roadmasks_dir = os.path.join(args.project_dir, "camera_calibration/rectified/roadmasks")
    model_dir = os.path.join(args.project_dir, "camera_calibration/rectified/sparse")

    # Read colmap cameras, images and points
    start_time = time.time()
    cameras, images_metas, points3d_in = read_model(model_dir)
    # calcuate camera K
    print("Calculating camera intrinsics...")
    camerasK = {}
    for camera_id, camera in cameras.items():
        fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
        camerasK[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    # calculate image masks for each images
    print("Loading road masks for images...")
    image_roadmasks = {}
    for image_id, image_meta in tqdm.tqdm(images_metas.items()):
        mask_path = os.path.join(roadmasks_dir, image_meta.name.split('.')[0] + '.png')
        if os.path.exists(mask_path):
            image_roadmasks[image_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            image_roadmasks[image_id] = None
            print(f"Warning: roadmask for image {image_meta.name} not found at {mask_path}")
    
    print("Extracting road points from SfM points...")
    onroad_points_xyz = []
    onroad_points_rgb = []
    for pid, pdata in tqdm.tqdm(points3d_in.items()):
        onroad_count = 0
        offroad_count = 0
        for image_id in pdata.image_ids:
            image_meta = images_metas[image_id]
            image_roadmask = image_roadmasks[image_id]
            if image_roadmask is None:
                continue
            qvec = image_meta.qvec
            tvec = image_meta.tvec
            cameraK = camerasK[image_meta.camera_id]
            u, v = project_point(pdata.xyz, qvec, tvec, cameraK)
            if 0 <= v < image_roadmask.shape[0] and 0 <= u < image_roadmask.shape[1]:
                if image_roadmask[v, u] > 127:
                    onroad_count += 1
                else:
                    offroad_count += 1
            # if contains any offroad projection, skip this point
            if offroad_count > 0:
                onroad_count = 0
                break
        if onroad_count > 1:
            onroad_points_xyz.append(pdata.xyz)
            onroad_points_rgb.append(pdata.rgb / 255.0)
    print(f"Total {len(onroad_points_xyz)} road points extracted from {len(points3d_in)} points.")
    onroad_points_xyz = np.array(onroad_points_xyz)
    onroad_points_rgb = np.array(onroad_points_rgb)
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'r', 'g', 'b']]
    onroad_points_attributes = np.concatenate((onroad_points_xyz, onroad_points_rgb), axis=1)
    elements = np.empty(onroad_points_xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, onroad_points_attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_path = os.path.join(model_dir, f"roadpoints_raw.ply")
    PlyData([el]).write(ply_path)
    
    #pcd = o3d.io.read_point_cloud(ply_path, format='ply')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(onroad_points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(onroad_points_rgb)
    # remove outliers: with radius 0.5m, at least 5 points
    pcd_clean, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.5)
    # further remove z outliers in local neighborhood
    pcd_clean, ind = remove_z_outliers(pcd_clean, radius=1.0, z_thresh=0.1)
    print(f"Cleaned road points from {len(pcd.points)} to {len(pcd_clean.points)} points.")
    onroad_points_xyz_clean = np.asarray(pcd_clean.points)
    onroad_points_rgb_clean = np.asarray(pcd_clean.colors)
    onroad_points_attributes = np.concatenate((onroad_points_xyz_clean, onroad_points_rgb_clean), axis=1)
    elements = np.empty(onroad_points_xyz_clean.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, onroad_points_attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_path = os.path.join(model_dir, f"roadpoints.ply")
    PlyData([el]).write(ply_path)

    global_end = time.time()
    print(f"process road sfm took {global_end - start_time} seconds.")
