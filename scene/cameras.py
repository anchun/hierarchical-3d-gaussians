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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import cv2

from utils.general_utils import PILtoTorch

import torch
import torch.nn.functional as F

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, primx, primy, image, alpha_mask,
                 invdepthmap, invdepthmap_npy,
                 image_name, uid, data_device = "cuda",
                 train_test_exp=False, is_test_dataset=False, is_test_view=False, is_novel_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.primx = primx
        self.primy = primy
        self.image_name = image_name
        self.is_novel_view = is_novel_view

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            if alpha_mask is not None:
                self.alpha_mask = PILtoTorch(alpha_mask, resolution)
            elif resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

            if train_test_exp and is_test_view:
                if is_test_dataset:
                    self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]

            if self.alpha_mask is not None:
                self.original_image *= self.alpha_mask
        else:
            self.original_image = None
            self.alpha_mask = None
            self.image_width = resolution[0]
            self.image_height = resolution[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None and depth_params is not None and depth_params["scale"] > 0:
            invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
            invdepthmapScaled = cv2.resize(invdepthmapScaled, resolution)
            invdepthmapScaled[invdepthmapScaled < 0] = 0
            if invdepthmapScaled.ndim != 2:
                invdepthmapScaled = invdepthmapScaled[..., 0]
            self.invdepthmap = torch.from_numpy(invdepthmapScaled[None]).to(self.data_device)

            if self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            else:
                self.depth_mask = torch.ones_like(self.invdepthmap > 0)
            
            if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]: 
                self.depth_mask *= 0
            else:
                self.depth_reliable = True

        if invdepthmap_npy is not None:
            self.invdepthmap_npy = torch.from_numpy(invdepthmap_npy).to(self.data_device)
            if self.depth_mask is None and self.alpha_mask is not None:
                self.depth_mask = self.alpha_mask.clone()
            self.depth_reliable = True
        else:
            self.invdepthmap_npy = None

        self.zfar = 1000.0
        self.znear = 0.01

        self.world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).to(self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, primx = primx, primy=primy).transpose(0,1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[3, :3].to(self.data_device)
    def shallow_clone(self):
        # clone without copying image/depth data
        new_cam = Camera(
            resolution=(self.image_width, self.image_height),
            colmap_id=self.colmap_id,
            R=self.R,
            T=self.T,
            FoVx=self.FoVx,
            FoVy=self.FoVy,
            depth_params={},
            primx=self.primx,
            primy=self.primy,
            image=None,
            alpha_mask=None,
            invdepthmap=None,
            invdepthmap_npy=None,
            image_name=self.image_name,
            uid=self.uid,
            data_device=self.data_device.type,
            train_test_exp=False,
            is_test_dataset=False,
            is_test_view=False,
            is_novel_view=self.is_novel_view
        )
        new_cam.world_view_transform = new_cam.world_view_transform.to(self.world_view_transform.device)
        new_cam.projection_matrix = new_cam.projection_matrix.to(self.world_view_transform.device)
        new_cam.full_proj_transform = new_cam.full_proj_transform.to(self.world_view_transform.device)
        new_cam.camera_center = new_cam.camera_center.to(self.world_view_transform.device)
        return new_cam

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


