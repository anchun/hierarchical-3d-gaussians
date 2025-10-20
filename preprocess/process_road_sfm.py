import argparse
import numpy as np
from read_write_model import *
import pyquaternion
import argparse
import os, time
from plyfile import PlyData, PlyElement
import open3d as o3d
from scipy.interpolate import griddata
from sklearn.neighbors import KDTree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
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

def alpha_shape_and_cameras_world_to_polygon(pcd, cameras_world, alpha=0.5):
    """
    将点云生成的 alpha-shape 网格和相机位置结合，转换为二维 Shapely 多边形
    """
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    vertices = np.asarray(mesh.vertices)[:, :2]
    triangles = np.asarray(mesh.triangles, dtype=int)
    
    # 将每个三角形转换为 Polygon 并进行合并
    polys = [Polygon(vertices[t]) for t in triangles]
    CAM_EXT = 0.5 # 以相机为中心的边长1米的正方形区域
    for cam in cameras_world:
        coords = ((cam[0] - CAM_EXT, cam[1] - CAM_EXT), (cam[0] + CAM_EXT, cam[1] - CAM_EXT),(cam[0] + CAM_EXT, cam[1] + CAM_EXT), (cam[0] - CAM_EXT, cam[1] + CAM_EXT), (cam[0] - CAM_EXT, cam[1] - CAM_EXT))
        polys.append(Polygon(coords))
    merged = unary_union(polys)  # 自动合并成单个或多个 Polygon
    return merged, mesh

def densify_road_with_alpha(pcd, cameras_world, alpha=0.5, resolution=0.1, interp_method='cubic'):
    """
    基于 α-shape 约束的路面点云插值算法
    输入：
        pcd: Open3D 点云对象 (稀疏路面点)
        cameras_world: 相机在世界坐标系下的位置列表
        alpha: α-shape 参数，越小边界越贴合
        resolution: 网格分辨率（单位：米）
        interp_method: 插值方式 ['linear', 'nearest', 'cubic']
    输出：
        dense_pcd: 插值后的密集路面点云 (Open3D 对象)
        mesh: 生成的 α-shape 网格 (Open3D 对象)
    """
    print("计算 α-shape 网格...")
    polygon, mesh = alpha_shape_and_cameras_world_to_polygon(pcd, cameras_world, alpha)
    prepared_poly = prep(polygon)  # 加速点在多边形内判断

    # 提取原始点坐标
    pts = np.asarray(pcd.points)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    # 生成插值网格
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    gx = np.arange(x_min, x_max, resolution)
    gy = np.arange(y_min, y_max, resolution)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    print("判断网格点是否在 α-shape 和孔洞 内部...")
    def inside_mask_func(px, py):
        return prepared_poly.contains(Point(px, py)) or \
                ( (prepared_poly.contains(Point(px - 1, py)) or prepared_poly.contains(Point(px - 2, py))) and \
                    (prepared_poly.contains(Point(px + 1, py)) or prepared_poly.contains(Point(px + 2, py))) ) or \
                ( (prepared_poly.contains(Point(px, py - 1)) or prepared_poly.contains(Point(px, py - 2))) and \
                    (prepared_poly.contains(Point(px, py + 1)) or prepared_poly.contains(Point(px, py + 2))) ) or \
                ( (prepared_poly.contains(Point(px-0.707, py-0.707)) or prepared_poly.contains(Point(px-1.414, py-1.414))) and \
                    (prepared_poly.contains(Point(px+0.707, py+0.707)) or prepared_poly.contains(Point(px+1.414, py+1.414))) )
    inside_mask = np.array([inside_mask_func(px, py)  for px, py in grid_xy])

    query_xy = grid_xy[inside_mask]

    print("进行高度插值...")
    grid_z = griddata((x, y), z, (query_xy[:, 0], query_xy[:, 1]), method=interp_method)
    valid = ~np.isnan(grid_z)
    grid_rgb = griddata((x, y), np.asarray(pcd.colors), (query_xy[:, 0], query_xy[:, 1]), method='nearest')

    dense_points = np.column_stack([query_xy[valid], grid_z[valid]])

    dense_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_points))
    dense_pcd.colors = o3d.utility.Vector3dVector(grid_rgb[valid])
    print(f"生成密集路面点云，共 {len(dense_points)} 个点。")

    return dense_pcd, mesh

def compute_z_diff(pcd1, pcd2):
    """
    计算点云A中每个点与点云B中最近(x,y)点的z差。
    参数:
        pc_a: np.ndarray, shape (N, 3) -> 点云A (x, y, z)
        pc_b: np.ndarray, shape (M, 3) -> 点云B (x, y, z)
    返回:
        z_diff: np.ndarray, shape (N,) -> 每个点的 z 差值
        idx_b: np.ndarray, shape (N,) -> 匹配到的B中索引
        dist_xy: np.ndarray, shape (N,) -> (x,y)距离
    """
    pc_a = np.asarray(pcd1.points)
    pc_b = np.asarray(pcd2.points)

    # 提取平面坐标
    xy_a = pc_a[:, :2]
    xy_b = pc_b[:, :2]

    # 构建KDTree，只在x,y平面上
    tree = KDTree(xy_b)

    # 查找A中每个点在B中的最近邻
    dist_xy, idx_b = tree.query(xy_a, k=1)
    idx_b = idx_b.squeeze()
    dist_xy = dist_xy.squeeze()

    # 获取对应的z值
    z_a = pc_a[:, 2]
    z_b = pc_b[idx_b, 2]

    # 计算z差
    z_diff = z_a - z_b

    return z_diff, dist_xy

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
    print("Step0: read colmap model and road masks...")
    cameras, images_metas, points3d_in = read_model(model_dir)
    # calcuate camera K
    print("Calculating camera intrinsics...")
    camerasK = {}
    for camera_id, camera in cameras.items():
        fx, fy, cx, cy = camera.params[0], camera.params[1], camera.params[2], camera.params[3]
        camerasK[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    mid_camera_id = (len(cameras) - 1) // 2
    opencv2camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    location_offset = [-3, -2, -1, 0, 1, 2]
    # calculate image masks for each images
    print("Loading camera poses and road masks for images...")
    image_roadmasks = {}
    cameras_world = []
    for image_id, image_meta in tqdm.tqdm(images_metas.items()):
        w2c = np.eye(4)
        w2c[:3, :3] = pyquaternion.Quaternion(image_meta.qvec).rotation_matrix
        w2c[:3, 3] = np.array(image_meta.tvec)
        camera_pose = np.linalg.inv(opencv2camera @ w2c)
        if mid_camera_id == image_meta.camera_id:
            for offset in location_offset:
                cameras_world.append(camera_pose[:3, 3] + camera_pose[:3, 0] * offset)
        else:
            cameras_world.append(camera_pose[:3, 3])
            
        mask_path = os.path.join(roadmasks_dir, image_meta.name.split('.')[0] + '.png')
        if os.path.exists(mask_path):
            image_roadmasks[image_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            image_roadmasks[image_id] = None
            print(f"Warning: roadmask for image {image_meta.name} not found at {mask_path}")
    
    print("Step1: Extracting road points from SfM points...")
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
    
    print("Step2: Cleaning road points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(onroad_points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(onroad_points_rgb)
    # remove outliers: with radius 0.5m, at least 5 points
    pcd_clean, ind = pcd.remove_radius_outlier(nb_points=5, radius=0.5)
    # further remove z outliers in local neighborhood
    pcd_clean, ind = remove_z_outliers(pcd_clean, radius=1.0, z_thresh=0.1)
    print(f"Cleaned road points from {len(pcd.points)} to {len(pcd_clean.points)} points.")
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'r', 'g', 'b']]
    onroad_points_xyz = np.asarray(pcd_clean.points)
    onroad_points_rgb = np.asarray(pcd_clean.colors)
    onroad_points_attributes = np.concatenate((onroad_points_xyz, onroad_points_rgb), axis=1)
    elements = np.empty(onroad_points_xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, onroad_points_attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_path = os.path.join(model_dir, f"roadpoints.ply")
    PlyData([el]).write(ply_path)
    
    print("Step3: densify road points...")
    pcd = pcd_clean
    #pcd = o3d.io.read_point_cloud(os.path.join(model_dir, f"roadpoints.ply"), format='ply')
    dense_pcd, _ = densify_road_with_alpha(
        pcd, cameras_world, alpha=2.0, resolution=0.1, interp_method='linear'
    )
    onroad_points_xyz = np.asarray(dense_pcd.points)
    onroad_points_rgb = np.asarray(dense_pcd.colors)
    onroad_points_attributes = np.concatenate((onroad_points_xyz, onroad_points_rgb), axis=1)
    elements = np.empty(onroad_points_xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, onroad_points_attributes))
    el = PlyElement.describe(elements, 'vertex')
    ply_path = os.path.join(model_dir, f"roadpoints_dense.ply")
    PlyData([el]).write(ply_path)
    
    print("Step4: compute statistics between original and dense road points...")
    z_diff, dist_xy = compute_z_diff(pcd, dense_pcd)
    print("平均XY距离:", np.mean(dist_xy), "平均Z差:", np.mean(z_diff))

    global_end = time.time()
    print(f"process road sfm took {global_end - start_time} seconds.")
