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
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import pyquaternion
import os, sys
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, sfm_points, is_test_dataset = False, is_novel_view = False):
    if not is_novel_view:
        image = Image.open(cam_info.image_path)

        if cam_info.mask_path != "":
            try:
                alpha_mask = Image.open(cam_info.mask_path).convert("L")
                if cam_info.mask2_path != "":
                    alpha_mask2 = Image.open(cam_info.mask2_path).convert("L")
                    mask1 = np.array(alpha_mask)
                    mask2 = np.array(alpha_mask2)
                    intersection = np.logical_and(mask1 > 128, mask2 > 128).astype(np.uint8) * 255
                    alpha_mask = Image.fromarray(intersection, mode="L")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                raise
        else:
            alpha_mask = None
        
        if cam_info.depth_path != "":
            try:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)
            except FileNotFoundError:
                print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
                raise
            except IOError:
                print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
                raise
        else:
            invdepthmap = None

        if cam_info.depth_npy_path is not None and cam_info.depth_npy_path != "":
            try:
                invdepthmap_npy = np.load(cam_info.depth_npy_path)
            except FileNotFoundError:
                print(f"Error: The depth file at path '{cam_info.depth_npy_path}' was not found.")
                raise
            except IOError:
                print(f"Error: Unable to open the image file '{cam_info.depth_npy_path}'. It may be corrupted or an unsupported format.")
                raise
            except Exception as e:
                print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_npy_path}: {e}")
                raise
        else:
            invdepthmap_npy = None
            
        if invdepthmap is None and invdepthmap_npy is None and sfm_points.shape[0] > 0:
            # calculate a rough invdepthmap from sfm points projection
            K = np.array([[fov2focal(cam_info.FovX, cam_info.width), 0, cam_info.primx * cam_info.width], [0, fov2focal(cam_info.FovY, cam_info.height), cam_info.primy * cam_info.height], [0, 0, 1]])
            points_cam = cam_info.R.T @ sfm_points.T + cam_info.T.reshape(3, 1)
            points_proj = (K @ points_cam).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[2:3, :].T  # (M,1)
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < cam_info.width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < cam_info.height)
                & (depths[:, 0] > 0)
                & (depths[:, 0] < 50)
            )
            points = points[selector]
            depths = 1.0 / depths[selector]
            invdepthmap_npy = np.concatenate((points, depths), axis=1).astype(np.float32)

        orig_w, orig_h = image.size

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 2560:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1080p pixels width), rescaling to 1080p.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 2560
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))
    else:
        resolution = (int(cam_info.width / resolution_scale), int(cam_info.height / resolution_scale))
        image = Image.open(cam_info.ref_image_path)
        alpha_mask = None
        invdepthmap = None
        invdepthmap_npy = None

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  primx=cam_info.primx, primy=cam_info.primy,
                  image=image, alpha_mask=alpha_mask, invdepthmap=invdepthmap,invdepthmap_npy=invdepthmap_npy,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test, is_novel_view = is_novel_view)


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

import torch

class CameraDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_cam_infos, sfm_point_cloud, args, resolution_scales, is_test, is_novel_view = False):
        'Initialization'
        self.resolution_scales = resolution_scales
        self.list_cam_infos = list_cam_infos
        self.sfm_point_cloud = sfm_point_cloud
        self.args = args
        self.args.data_device = 'cpu'
        self.is_test = is_test
        self.is_novel_view = is_novel_view

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_cam_infos)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Select sample
        info = self.list_cam_infos[index]
        X = loadCam(self.args, index, info, self.resolution_scales, sfm_points =self.sfm_point_cloud.points, is_test_dataset = self.is_test, is_novel_view = self.is_novel_view)

        return X
  
