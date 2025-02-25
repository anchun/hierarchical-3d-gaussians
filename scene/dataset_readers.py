#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
import joblib
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    primx:float
    primy:float
    depth_params: dict
    image_path: str
    mask_path: str
    depth_path: str
    depth_npy_path: str
    image_name: str
    width: int
    height: int
    is_test: bool
    metadata: dict
    cam_idx: int = 0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    scene_meta: dict

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.quantile(dist, 0.9)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list, use_npy_depth=False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            primx = float(intr.params[1]) / width
            primy = float(intr.params[2]) / height
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            primx = float(intr.params[2]) / width
            primy = float(intr.params[3]) / height
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.jpg")
            image_name = f"{extr.name[:-n_remove]}.jpg"
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.png")
            image_name = f"{extr.name[:-n_remove]}.png"

        mask_path = os.path.join(masks_folder, f"{extr.name[:-n_remove]}.png") if masks_folder != "" else ""
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""
        if use_npy_depth:
            depth_path = ""
            depth_npy_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.npy") if depths_folder != "" else ""
        else:
            depth_npy_path = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, depth_npy_path=depth_npy_path, image_name=image_name, 
                              width=width, height=height, is_test=image_name in test_cam_names_list, metadata=metadata)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def name_2_frame_id(name):
    #'cam_0/000009.png'
    frame_id = name.split('/')[1].split('.')[0]
    return int(frame_id)


def readNOTRCameras(cam_pose_info_in_colmap, cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list, use_npy_depth=False):
    cam_infos = []
    for idx, image_id in enumerate(cam_pose_info_in_colmap):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_pose_info_in_colmap)))
        sys.stdout.flush()
        raw_cam_define = cam_pose_info_in_colmap[image_id]
        camera_id = raw_cam_define.camera_id
        metadata = {}
        metadata['frame_id'] = name_2_frame_id(raw_cam_define.name)
        metadata['extrinsic'] = cam_extrinsics[camera_id]
        # metadata['timestamp'] = cams_timestamps[i]
        intr = cam_intrinsics[camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(raw_cam_define.qvec))
        T = np.array(raw_cam_define.tvec)
        RT = np.zeros((4, 4))
        RT[:3, :3] = R.T  # R 转置得到 RT 的左上角 3x3 子矩阵
        RT[:3, 3] = T  # T 作为 RT 的前三行第四列元素
        RT[3, :] = [0, 0, 0, 1]  # 补充 RT 的第四行
        c2w = np.linalg.inv(RT)
        metadata['ego_pose'] = c2w.dot(np.linalg.inv(metadata['extrinsic']))

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            primx = float(intr.params[1]) / width
            primy = float(intr.params[2]) / height
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            primx = float(intr.params[2]) / width
            primy = float(intr.params[3]) / height
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(raw_cam_define.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[raw_cam_define.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, raw_cam_define.name)
        image_name = raw_cam_define.name
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{raw_cam_define.name[:-n_remove]}.jpg")
            image_name = f"{raw_cam_define.name[:-n_remove]}.jpg"
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{raw_cam_define.name[:-n_remove]}.png")
            image_name = f"{raw_cam_define.name[:-n_remove]}.png"

        mask_path = os.path.join(masks_folder, f"{raw_cam_define.name[:-n_remove]}.png") if masks_folder != "" else ""
        depth_path = os.path.join(depths_folder, f"{raw_cam_define.name[:-n_remove]}.png") if depths_folder != "" else ""
        if use_npy_depth:
            depth_path = ""
            depth_npy_path = os.path.join(depths_folder, f"{raw_cam_define.name[:-n_remove]}.npy") if depths_folder != "" else ""
        else:
            depth_npy_path = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, depth_npy_path=depth_npy_path, image_name=image_name,
                              width=width, height=height, is_test=image_name in test_cam_names_list, metadata=metadata)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if('red' in vertices):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.ones_like(positions) * 0.5
    if('nx' in vertices):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPt(xyz_path, rgb_path):
    positions_tensor = torch.jit.load(xyz_path).state_dict()['0']

    positions = positions_tensor.numpy()

    colors_tensor = torch.jit.load(rgb_path).state_dict()['0']
    if colors_tensor.size(0) == 0:
        colors_tensor = 255 * (torch.ones_like(positions_tensor) * 0.5)
    colors = (colors_tensor.float().numpy()) / 255.0
    normals = torch.Tensor([]).numpy()

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, masks, depths, eval, train_test_exp, llffhold=None, use_npy_depth=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "" and not use_npy_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    try:
        xyz_path = os.path.join(path, "sparse/0/xyz.pt")
        rgb_path = os.path.join(path, "sparse/0/rgb.pt")
        pcd = fetchPt(xyz_path, rgb_path)
    except:
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    masks_reading_dir = masks if masks == "" else os.path.join(path, masks)
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params, 
        images_folder=os.path.join(path, reading_dir), masks_folder=masks_reading_dir,
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list, use_npy_depth=use_npy_depth)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNOTRSceneInfo(project_dir, path, images, masks, depths, eval, train_test_exp, llffhold=None, use_npy_depth=False):
    """
    scene_meta：表示原始整个大场景的信息，dict类型
    {
        'num_frames': int
        'exts': ndarray(num_frames*num_cameras, 4,4)，相机外参，camera按0，1，2，3。。。对应各相机
        'ixts': ndarray(num_frames*num_cameras, 3,3)，相机内参，camera按0，1，2，3。。。对应各相机
        'poses': ndarray(num_frames*num_cameras, 4,4)，ego_poses，camera按0，1，2，3。。。对应各相机
        'c2ws: ndarray(num_frames*num_cameras, 4,4)，poses @ exts
        'obj_tracklets': ndarray(num_frames, max_obj, 8)，第二维为表示所有帧中出现最多动态物的动态物数量, 第三维依次为：obj_id,x,y,z,qw,qx,qy,qz（相对主车）
        'obj_info': dict，key为obj_id，value为obj_meta
                obj_meta:{
                    'track_id'：int obj_id
                    'class': str，见waymo_utils.py waymo_track2label
                    'class_label': int 见waymo_utils.py waymo_track2label
                    'width','height','length': float
                    'deformable': bool
                    'start_frame', 'end_frame': int
                    'start_timestamp', 'end_timestamp': float
                }
        'frames': int list, [num_cameras*num_frames]，每个相机的帧号，例：0,0,0,0,1,1,1,1,2,2,2,2.... （4个相机）
        'cams': int list, [num_cameras*num_frames]，N个相机重复，例：0,1,2,3,0,1,2,3,0,1,2,3...（4个相机）
        'frames_idx'：和frames相同
        'new_image_filenames' str list，[num_cameras*num_frames]，每个相机的图像文件相对路径
        'cams_timestamps' ndarray[num_cameras*num_frames]，每个相机的相对时间戳，第一帧不绝对等于0，而是接近0的值
        'tracklet_timestamps': ndarray[num_frames]，未知
        'obj_bounds'：未知
    }
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_pose_info_in_colmap = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_pose_info_in_colmap = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    scene_meta = joblib.load(os.path.join(project_dir, "scene_meta.bin"))
    cam_extrinsics = scene_meta['extrinsics'][:len(cam_intrinsics.keys())]
    # num_frames = scene_meta['num_frames']
    ego_pose_dir = os.path.join(project_dir, 'ego_pose')
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    ego_poses = []
    for ego_pose_path in ego_pose_paths:
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_poses.append(ego_frame_pose)
    if depths != "" and not use_npy_depth:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    try:
        xyz_path = os.path.join(path, "sparse/0/xyz.pt")
        rgb_path = os.path.join(path, "sparse/0/rgb.pt")
        pcd = fetchPt(xyz_path, rgb_path)
    except:
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)

    reading_dir = "images" if images == None else images
    masks_reading_dir = masks if masks == "" else os.path.join(path, masks)

    cam_infos_unsorted = readNOTRCameras(
        cam_pose_info_in_colmap=cam_pose_info_in_colmap,
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), masks_folder=masks_reading_dir,
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=[],
        use_npy_depth=use_npy_depth)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           scene_meta=scene_meta
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
    , "NOTR": readNOTRSceneInfo
}
