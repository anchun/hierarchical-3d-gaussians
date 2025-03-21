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

import json
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, quaternion_to_matrix, quaternion_raw_multiply, matrix_to_quaternion
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from gaussian_hierarchy._C import load_hierarchy, write_hierarchy
from scene.OurAdam import Adam
from scene.cameras import Camera
from scene.gaussian_model_actor import GaussianModelActor
from scene.actor_pose import ActorPose
import time
from scene.camera_pose_corretion import PoseCorrection

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        # Build object model
        self.obj_list = []
        if self.metadata is not None and 'obj_info' in self.metadata.keys() and len(self.metadata['obj_info'].keys()) > 0:
            obj_infos = self.metadata['obj_info']
            for object_id, obj_info in obj_infos.items():
                if obj_info['start_frame'] == obj_info['end_frame']:
                    continue
                obj_meta = obj_info.copy()
                obj_meta['object_id'] = object_id # 原obj_info中用track_id表示object_id
                obj_model = GaussianModelActor(obj_meta=obj_meta, num_frames=self.metadata['num_frames'],max_sh_degree=self.max_sh_degree,num_classes=self.num_classes)
                setattr(self, obj_model.get_modelname, obj_model)
                self.obj_list.append(obj_model)
            obj_tracklets = self.metadata['obj_tracklets']
            tracklet_timestamps = self.metadata['tracklet_timestamps']
            camera_timestamps = self.metadata['cams_timestamps']
            self.actor_pose = ActorPose(obj_tracklets, tracklet_timestamps, camera_timestamps, obj_infos)
        else:
            self.actor_pose = None

        if self.use_camera_pose_correction:
            self.pose_correction = PoseCorrection(self.num_camera_poses)
        else:
            self.pose_correction = None

    def __init__(self, args, scene_info, metadata: dict, num_camera_poses: int, num_classes: int=0, use_camera_pose_correction: bool=False, state_dict=None, saved_ply_folder=None):
        self.num_classes = num_classes
        self.use_camera_pose_correction = use_camera_pose_correction
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantic = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.nodes = None
        self.boxes = None
        self.num_camera_poses = num_camera_poses
        self.is_dynamic_object_enabled = False
        self.visible_objects = []

        self.pretrained_exposures = None

        self.skybox_points = 0
        self.skybox_locked = True

        self.metadata = metadata
        self.visible_dynamic_object_names = []
        self.inference_visible_replacements = {}
        self.setup_functions()
        if state_dict is not None:
            # 从checkpoint恢复训练时，需要restore全部状态
            # inference时，仅从state_dict加载actor_pose等参数，并且从ply中加载点云
            # 目前3dgs项目的存储分为ply和其他参数，比较混乱
            only_load_pose_weights = saved_ply_folder is not None
            self.restore(state_dict, training_args=None, only_pose_weights=only_load_pose_weights)
            if saved_ply_folder is not None:
                # inference时，从state_dict加载actor_pose等参数，并且从ply中加载点云
                iteration = state_dict['iteration']
                self.load_ply(saved_ply_folder)
        # elif args.pretrained:
        #     self.gaussians.create_from_pt(args.pretrained, self.cameras_extent)
        # elif create_from_hier:
        #     self.gaussians.create_from_hier(args.hierarchy, self.cameras_extent, args.scaffold_file)
        else:
            # 初始化训练时，加载初始点云
            self.create_from_pcd(scene_info.point_cloud,
                                           scene_info.train_cameras,
                                           scene_info.nerf_normalization["radius"],
                                           args.skybox_num,
                                           args.scaffold_file,
                                           args.bounds_file,
                                           args.skybox_locked)

    def disable_dynamic_objects(self):
        self.is_dynamic_object_enabled = False

    def enable_dynamic_objects(self):
        self.is_dynamic_object_enabled = True

    def capture(self, only_pose_weights=False):
        state_dict = {}
        if only_pose_weights:
            if self.actor_pose is not None:
                state_dict['actor_pose'] = self.actor_pose.save_state_dict()

            if self.pose_correction is not None:
                state_dict['pose_correction'] = self.pose_correction.save_state_dict(True)
        else:
            state_dict['bkgd'] = (
                        self.active_sh_degree,
                        self._xyz,
                        self._features_dc,
                        self._features_rest,
                        self._scaling,
                        self._rotation,
                        self._opacity,
                        self.max_radii2D,
                        self.xyz_gradient_accum,
                        self.denom,
                        self.optimizer.state_dict(),
                        self.spatial_lr_scale,
                    )
            for dynamic_obj in self.obj_list:
                model_name = dynamic_obj.get_modelname
                state_dict[model_name] = dynamic_obj.save_state_dict()

            if self.actor_pose is not None:
                state_dict['actor_pose'] = self.actor_pose.save_state_dict()

            if self.pose_correction is not None:
                state_dict['pose_correction'] = self.pose_correction.save_state_dict(False)

            state_dict['bkgd_optimizer'] = self.optimizer.state_dict()
            state_dict['bkgd_exposure_optimizer'] = self.exposure_optimizer.state_dict()
        return state_dict
    
    def restore(self, state_dict, training_args=None, only_pose_weights=False):
        if only_pose_weights:
            if self.actor_pose is not None:
                self.actor_pose.load_state_dict(state_dict['actor_pose'])

            if self.pose_correction is not None:
                self.pose_correction.load_state_dict(state_dict['pose_correction'])
        else:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            opt_dict,
            self.spatial_lr_scale) = state_dict['bkgd']
            if training_args is not None:
                self.training_setup(training_args)
            else:
                pass # inference

            for dynamic_obj in self.obj_list:
                dynamic_obj.restore(state_dict[dynamic_obj.get_modelname])

            if self.actor_pose is not None:
                self.actor_pose.load_state_dict(state_dict['actor_pose'])

            if self.pose_correction is not None:
                self.pose_correction.load_state_dict(state_dict['pose_correction'])

    def restore_training_status(self, state_dict):
        self.optimizer.load_state_dict(state_dict['bkgd_optimizer'])
        self.exposure_optimizer.load_state_dict(state_dict['bkgd_exposure_optimizer'])
        for dynamic_obj in self.obj_list:
            dynamic_obj.restore_training_status(state_dict[dynamic_obj.get_modelname])

    def set_visible_dynamic_object_names(self, visible_model_names):
        self.visible_dynamic_object_names = visible_model_names

    def replace_inference_models(self, replacements):
        self.inference_visible_replacements = replacements

    def parse_camera(self, camera: Camera):
        self.viewpoint_camera = camera
        self.current_frame = camera.metadata['frame_id']
        self.current_camera_idx = camera.cam_idx

        self.visible_objects = []
        self.num_gaussians = 0
        timestamp = camera.metadata['timestamp']

        # 训练时，每次forward需要控制是否更新动态对象，见train_single.py
        # 推理时，可以控制哪里动态对象可见
        if type(self.visible_dynamic_object_names) == str and self.visible_dynamic_object_names == 'all':
            candidates = self.obj_list
        elif type(self.visible_dynamic_object_names) == str and self.visible_dynamic_object_names == 'none':
            candidates = []
        else:
            candidates = []
            for obj_model in self.obj_list:
                if obj_model.get_modelname in self.visible_dynamic_object_names:
                    candidates.append(obj_model)

        for obj_model in candidates:
            start_timestamp, end_timestamp = obj_model.start_timestamp, obj_model.end_timestamp
            if timestamp >= start_timestamp and timestamp <= end_timestamp and obj_model.get_xyz.size(0) > 0 and len(obj_model.get_xyz.shape) == 2: # 会有xyz.shape3维的情况。TODO 待查
                self.visible_objects.append(obj_model)
                num_gaussians_obj = obj_model.get_xyz.shape[0]
                self.num_gaussians += num_gaussians_obj

        self.graph_gaussian_range = dict()
        idx = 0
        num_gaussians_bkgd = self._xyz.shape[0]
        self.graph_gaussian_range['background'] = [idx, idx+num_gaussians_bkgd-1]
        idx += num_gaussians_bkgd

        for obj_model in self.visible_objects:
            obj_name = obj_model.get_modelname
            if obj_model.get_modelname in self.inference_visible_replacements.keys():
                # TODO 这是测试渲染时的编辑操作时临时使用的
                replace_to_model = getattr(self, self.inference_visible_replacements[obj_model.get_modelname])
                num_gaussians_obj = replace_to_model.get_xyz.shape[0]
            else:
                num_gaussians_obj = obj_model.get_xyz.shape[0]
            self.graph_gaussian_range[obj_name] = [idx, idx+num_gaussians_obj-1]
            idx += num_gaussians_obj

        if len(self.visible_objects) > 0:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_model in enumerate(self.visible_objects):
                object_id = obj_model.object_id
                if obj_model.get_modelname in self.inference_visible_replacements.keys():
                # TODO 这是测试渲染时的编辑操作时临时使用的
                    target_name = self.inference_visible_replacements[obj_model.get_modelname]
                    target_obj = getattr(self, target_name)
                    self.visible_objects[i] = target_obj
                    obj_model = target_obj

                obj_rot = self.actor_pose.get_tracking_rotation(object_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(object_id, self.viewpoint_camera)
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
                
                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)
                
                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)
            
            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)  
            
            # self.flip_mask = []
            # for obj_name in self.visible_objects:
            #     obj_model: GaussianModelActor = getattr(self, obj_name)
            #     if obj_model.deformable or self.flip_prob == 0:
            #         flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
            #     else:
            #         flip_mask = torch.rand_like(obj_model.get_xyz[:, 0]) < self.flip_prob
            #     self.flip_mask.append(flip_mask)
            # self.flip_mask = torch.cat(self.flip_mask, dim=0)

    @property
    def get_scaling(self):
        scalings = []
        scaling_bkgd = self.scaling_activation(self._scaling)
        scalings.append(scaling_bkgd)
        for obj_model in self.visible_objects:
            scaling = obj_model.get_scaling
            scalings.append(scaling)
        scalings = torch.cat(scalings, dim=0)
        return scalings

    @property
    def get_self_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_bg_scaling(self):
        return self.scaling_activation(self._scaling)


    @property
    def get_rotation(self):
        rotations = []
        rotations_bkgd = self.rotation_activation(self._rotation)
        if self.use_camera_pose_correction:
            rotations_bkgd = self.pose_correction.correct_gaussian_rotation(self.current_camera_idx, rotations_bkgd)
        rotations.append(rotations_bkgd)

        if len(self.visible_objects) > 0:
            rotations_local = []
            for i, obj_model in enumerate(self.visible_objects):
                rotation_local = obj_model.get_rotation
                rotations_local.append(rotation_local)

            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            # rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations

    @property
    def get_xyz(self):
        xyzs = []
        if self.use_camera_pose_correction:
            xyzs.append(self.pose_correction.correct_gaussian_xyz(self.current_camera_idx, self._xyz))
        else:
            xyzs.append(self._xyz)
        if len(self.visible_objects) > 0:
            xyzs_local = []

            for i, obj_model in enumerate(self.visible_objects):
                xyz_local = obj_model.get_xyz
                xyzs_local.append(xyz_local)

            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            # xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj)
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features_bkgd = torch.cat((features_dc, features_rest), dim=1)

        features = []
        features.append(features_bkgd)
        for obj_model in self.visible_objects:
            feature_obj = obj_model.get_features_fourier(self.current_frame)
            features.append(feature_obj)
        features = torch.cat(features, dim=0)
        return features

    @property
    def get_semantic(self):
        semantics = []
        if True: #self.semantic_mode == 'logits':
            semantic_bkgd = self._semantic
        else: # 'probabilities':
            semantic_bkgd = torch.nn.functional.softmax(self._semantic, dim=1)
        semantics.append(semantic_bkgd)
        for obj_model in self.visible_objects:
            semantic = obj_model.get_semantic
            semantics.append(semantic)
        semantics = torch.cat(semantics, dim=0)
        return semantics

    @property
    def get_opacity(self):
        opacity_bkgd = self.opacity_activation(self._opacity)
        opacities = []
        opacities.append(opacity_bkgd)
        for obj_model in self.visible_objects:
            opacity = obj_model.get_opacity
            opacities.append(opacity)
        opacities = torch.cat(opacities, dim=0)
        return opacities

    @property
    def get_self_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        return self._exposure[self.exposure_mapping[image_name]]
        # return self._exposure

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
            self, 
            point_clouds : dict,
            cam_infos : int,
            spatial_lr_scale : float,
            skybox_points: int,
            scaffold_file: str,
            bounds_file: str,
            skybox_locked: bool):
        for obj_model in self.obj_list:
            obj_model.create_from_pcd(point_clouds)

        self.spatial_lr_scale = spatial_lr_scale

        pcd = point_clouds['bkgd']
        xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        
        minimum,_ = torch.min(xyz, axis=0)
        maximum,_ = torch.max(xyz, axis=0)
        mean = 0.5 * (minimum + maximum)

        self.skybox_locked = skybox_locked
        if scaffold_file != "" and skybox_points > 0:
            print(f"Overriding skybox_points: loading skybox from scaffold_file: {scaffold_file}")
            skybox_points = 0
        if skybox_points > 0:
            self.skybox_points = skybox_points
            radius = torch.linalg.norm(maximum - mean)

            theta = (2.0 * torch.pi * torch.rand(skybox_points, device="cuda")).float()
            phi = (torch.arccos(1.0 - 1.4 * torch.rand(skybox_points, device="cuda"))).float()
            skybox_xyz = torch.zeros((skybox_points, 3))
            skybox_xyz[:, 0] = radius * 10 * torch.cos(theta)*torch.sin(phi)
            skybox_xyz[:, 1] = radius * 10 * torch.sin(theta)*torch.sin(phi)
            skybox_xyz[:, 2] = radius * 10 * torch.cos(phi)
            skybox_xyz += mean.cpu()
            xyz = torch.concat((skybox_xyz.cuda(), xyz))
            fused_color = torch.concat((torch.ones((skybox_points, 3)).cuda(), fused_color))
            fused_color[:skybox_points,0] *= 0.7
            fused_color[:skybox_points,1] *= 0.8
            fused_color[:skybox_points,2] *= 0.95

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = RGB2SH(fused_color)
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(xyz), 0.0000001)
        if scaffold_file == "" and skybox_points > 0:
            dist2[:skybox_points] *= 10
            dist2[skybox_points:] = torch.clamp_max(dist2[skybox_points:], 10) 
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if scaffold_file == "" and skybox_points > 0:
            opacities = self.inverse_opacity_activation(0.02 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
            opacities[:skybox_points] = 0.7
        else: 
            opacities = self.inverse_opacity_activation(0.01 * torch.ones((xyz.shape[0], 1), dtype=torch.float, device="cuda"))
        semantics = torch.zeros((xyz.shape[0], self.num_classes), dtype=torch.float, device="cuda")

        features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        features_rest = features[:,:,1:].transpose(1, 2).contiguous()

        self.scaffold_points = None
        if scaffold_file != "": 
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold, semantics_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()
            semantics_scaffold = torch.from_numpy(semantics_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                skybox_points = int(f.readline())

            self.skybox_points = skybox_points
            with open(os.path.join(bounds_file, "center.txt")) as centerfile:
                with open(os.path.join(bounds_file, "extent.txt")) as extentfile:
                    centerline = centerfile.readline()
                    extentline = extentfile.readline()

                    c = centerline.split(' ')
                    e = extentline.split(' ')
                    center = torch.Tensor([float(c[0]), float(c[1]), float(c[2])]).cuda()
                    extent = torch.Tensor([float(e[0]), float(e[1]), float(e[2])]).cuda()

            distances1 = torch.abs(scaffold_xyz.cuda() - center)
            selec = torch.logical_and(
                torch.max(distances1[:,0], distances1[:,1]) > 0.5 * extent[0],
                torch.max(distances1[:,0], distances1[:,1]) < 1.5 * extent[0])
            selec[:skybox_points] = True

            self.scaffold_points = selec.nonzero().size(0)

            xyz = torch.concat((scaffold_xyz.cuda()[selec], xyz))
            features_dc = torch.concat((features_dc_scaffold.cuda()[selec,0:1,:], features_dc))

            rest_dim = 3 if self.max_sh_degree == 1 else 15
            if self.max_sh_degree > 0:
                filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), rest_dim, 3))
                filler[:,0:3,:] = features_extra_scaffold.cuda()[selec,:,:]
            else:
                filler = torch.zeros((features_extra_scaffold.cuda()[selec,:,:].size(0), 0, 3))
            features_rest = torch.concat((filler.cuda(), features_rest))
            scales = torch.concat((scales_scaffold.cuda()[selec], scales))
            rots = torch.concat((rots_scaffold.cuda()[selec], rots))
            semantics = torch.concat((semantics_scaffold.cuda()[selec], semantics))
            opacities = torch.concat((opacities_scaffold.cuda()[selec], opacities))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._semantic = nn.Parameter(semantics.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))
        print("Number of points at initialisation : ", self._xyz.shape[0])

    def training_setup(self, training_args, our_adam=True):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': training_args.semantic_lr, "name": "semantic"},
        ]

        if our_adam:
            self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        else:
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if self.pretrained_exposures is None:
            self.exposure_optimizer = torch.optim.Adam([self._exposure])
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final, lr_delay_steps=training_args.exposure_lr_delay_steps, lr_delay_mult=training_args.exposure_lr_delay_mult, max_steps=training_args.iterations)
        for obj_model in self.obj_list:
            obj_model.training_setup(training_args)

        if self.actor_pose is not None:
            self.actor_pose.training_setup()

        if self.pose_correction is not None:
            self.pose_correction_scheduler_args = get_expon_lr_func(
                    # warmup_steps=0,
                    lr_init=5e-6,
                    lr_final=1e-6,
                    max_steps=training_args.iterations,
                )
            self.pose_correction.training_setup(training_args.iterations)
       
    def load_ply_file(self, path, degree):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        semantic_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("semantic_")]
        semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((xyz.shape[0], len(semantic_names)))
        if self.num_classes > 0:
            for idx, attr_name in enumerate(semantic_names):
               semantic[:, idx] = np.asarray(plydata.elements[0][attr_name])

        return xyz, features_dc, features_extra, opacities, scales, rots, semantic


    def create_from_hier(self, path, spatial_lr_scale : float, scaffold_file : str):
        self.spatial_lr_scale = spatial_lr_scale

        xyz, shs_all, alpha, scales, rots, nodes, boxes, semantics = load_hierarchy(path)

        base = os.path.dirname(path)

        try:
            with open(os.path.join(base, "anchors.bin"), mode='rb') as f:
                bytes = f.read()
                int_val = int.from_bytes(bytes[:4], "little", signed="False")
                dt = np.dtype(np.int32)
                vals = np.frombuffer(bytes[4:], dtype=dt) 
                self.anchors = torch.from_numpy(vals).long().cuda()
        except:
            print("WARNING: NO ANCHORS FOUND")
            self.anchors = torch.Tensor([]).long()

        #retrieve exposure
        exposure_file = os.path.join(base, "exposure.json")
        if os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)

            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
        else:
            print(f"No exposure to be loaded at {exposure_file}")
            self.pretrained_exposures = None

        #retrieve skybox
        self.skybox_points = 0         
        if scaffold_file != "":
            scaffold_xyz, features_dc_scaffold, features_extra_scaffold, opacities_scaffold, scales_scaffold, rots_scaffold, semantics_scaffold = self.load_ply_file(scaffold_file + "/point_cloud.ply", 1)
            scaffold_xyz = torch.from_numpy(scaffold_xyz).float()
            features_dc_scaffold = torch.from_numpy(features_dc_scaffold).permute(0, 2, 1).float()
            features_extra_scaffold = torch.from_numpy(features_extra_scaffold).permute(0, 2, 1).float()
            opacities_scaffold = torch.from_numpy(opacities_scaffold).float()
            scales_scaffold = torch.from_numpy(scales_scaffold).float()
            rots_scaffold = torch.from_numpy(rots_scaffold).float()
            semantics_scaffold = torch.from_numpy(semantics_scaffold).float()

            with open(scaffold_file + "/pc_info.txt") as f:
                    skybox_points = int(f.readline())

            self.skybox_points = skybox_points

        if self.skybox_points > 0:
            if scaffold_file != "":
                skybox_xyz, features_dc_sky, features_rest_sky, opacities_sky, scales_sky, rots_sky, semantics_sky = scaffold_xyz[:skybox_points], features_dc_scaffold[:skybox_points], features_extra_scaffold[:skybox_points], opacities_scaffold[:skybox_points], scales_scaffold[:skybox_points], rots_scaffold[:skybox_points], semantics_scaffold[:skybox_points]

            opacities_sky = torch.sigmoid(opacities_sky)
            xyz = torch.cat((xyz, skybox_xyz))
            alpha = torch.cat((alpha, opacities_sky))
            scales = torch.cat((scales, scales_sky))
            rots = torch.cat((rots, rots_sky))
            semantics = torch.cat((semantics, semantics_sky))
            filler = torch.zeros(features_dc_sky.size(0), 16, 3)
            filler[:, :1, :] = features_dc_sky
            filler[:, 1:4, :] = features_rest_sky
            shs_all = torch.cat((shs_all, filler))

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_all.cuda()[:,:1,:].requires_grad_(True))
        self._features_rest = nn.Parameter(shs_all.cuda()[:,1:16,:].requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self._semantic = nn.Parameter(semantics.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = torch.abs

        self.hierarchy_path = path

        self.nodes = nodes.cuda()
        self.boxes = boxes.cuda()

    def create_from_pt(self, path, spatial_lr_scale : float ):
        self.spatial_lr_scale = spatial_lr_scale

        xyz = torch.load(path + "/done_xyz.pt")
        shs_dc = torch.load(path + "/done_dc.pt")
        shs_rest = torch.load(path + "/done_rest.pt")
        alpha = torch.load(path + "/done_opacity.pt")
        scales = torch.load(path + "/done_scaling.pt")
        rots = torch.load(path + "/done_rotation.pt")

        self._xyz = nn.Parameter(xyz.cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(shs_dc.cuda().requires_grad_(True))
        self._features_rest = nn.Parameter(shs_rest.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(alpha.cuda().requires_grad_(True))
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        #self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def save_hier(self):
        write_hierarchy(self.hierarchy_path + "_opt",
                        self._xyz,
                        torch.cat((self._features_dc, self._features_rest), 1),
                        self.opacity_activation(self._opacity),
                        self._scaling,
                        self._rotation,
                        self.nodes,
                        self.boxes,
                        # self._semantic
                        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''

        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

        for obj_model in self.obj_list:
            obj_model.update_learning_rate(iteration)

        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)

        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)

    def update_optimizer(self):
        if self._opacity.grad != None:
            relevant = (self._opacity.grad.flatten() != 0).nonzero()
            relevant = relevant.flatten().long()
            self.optimizer.step(relevant)
            self.optimizer.zero_grad(set_to_none = True)

        for obj_model in self.obj_list:
            obj_model.update_optimizer()

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()

        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.num_classes > 0:
            for i in range(self._semantic.shape[1]):
               l.append('semantic_{}'.format(i))
        return l

    def save_pt(self, path):
        mkdir_p(path)

        torch.save(self._xyz.detach().cpu(), os.path.join(path, "done_xyz.pt"))
        torch.save(self._features_dc.cpu(), os.path.join(path, "done_dc.pt"))
        torch.save(self._features_rest.cpu(), os.path.join(path, "done_rest.pt"))
        torch.save(self._opacity.cpu(), os.path.join(path, "done_opacity.pt"))
        torch.save(self._scaling, os.path.join(path, "done_scaling.pt"))
        torch.save(self._rotation, os.path.join(path, "done_rotation.pt"))
        torch.save(self._semantic, os.path.join(path, "done_semantic.pt"))

        import struct
        def load_pt(path):
            xyz = torch.load(os.path.join(path, "done_xyz.pt")).detach().cpu()
            features_dc = torch.load(os.path.join(path, "done_dc.pt")).detach().cpu()
            features_rest = torch.load( os.path.join(path, "done_rest.pt")).detach().cpu()
            opacity = torch.load(os.path.join(path, "done_opacity.pt")).detach().cpu()
            scaling = torch.load(os.path.join(path, "done_scaling.pt")).detach().cpu()
            rotation = torch.load(os.path.join(path, "done_rotation.pt")).detach().cpu()
            semantic = torch.load(os.path.join(path, "done_semantic.pt")).detach().cpu()

            return xyz, features_dc, features_rest, opacity, scaling, rotation, semantic


        xyz, features_dc, features_rest, opacity, scaling, rotation, semantic = load_pt(path)

        my_int = xyz.size(0)
        with open(os.path.join(path, "point_cloud.bin"), 'wb') as f:
            f.write(struct.pack('i', my_int))
            f.write(xyz.numpy().tobytes())
            print(features_dc[0])
            print(features_rest[0])
            f.write(torch.cat((features_dc, features_rest), dim=1).numpy().tobytes())
            f.write(opacity.numpy().tobytes())
            f.write(scaling.numpy().tobytes())
            f.write(rotation.numpy().tobytes())
            f.write(semantic.numpy().tobytes())

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        bg_plydata_list = []

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        semantic = self._semantic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self.num_classes > 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, semantic), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)
        bg_plydata_list.append(el)

        for obj_model in self.obj_list:
            plydata = obj_model.make_ply()
            model_name = obj_model.get_modelname
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            PlyData([plydata]).write(os.path.join(path, model_name + "_point_cloud.ply"))
            print(f'==== {model_name} saved, num_points: {obj_model._xyz.size(0)}')

        PlyData(bg_plydata_list).write(os.path.join(path, "point_cloud.ply"))

    def reset_opacity(self):
        opacities_new = torch.cat((self._opacity[:self.skybox_points], inverse_sigmoid(torch.min(self.get_self_opacity[self.skybox_points:], torch.ones_like(self.get_self_opacity[self.skybox_points:])*0.01))), 0)
        #opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        for obj_model in self.obj_list:
            obj_model.reset_opacity()

    def load_ply(self, dir):
        for model in self.obj_list:
            model_name = model.get_modelname
            path = os.path.join(dir, model_name + "_point_cloud.ply")
            plydata_list = PlyData.read(path).elements
            plydata = plydata_list[0]
            # model_name = plydata.name[7:] # vertex_.....
            print('Loading model', model_name)
            model.load_ply(path=None, input_ply=plydata)

        xyz, features_dc, features_extra, opacities, scales, rots, semantic = self.load_ply_file(os.path.join(dir, "point_cloud.ply"), self.max_sh_degree)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
             "semantic": new_semantic
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")))
        #self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad * self.max_radii2D * torch.pow(self.get_self_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_self_opacity.flatten() > 0.15)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_self_scaling, dim=1).values > self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False

        stds = self.get_self_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_self_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_semantic)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_self_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_self_opacity.flatten() > 0.15)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_self_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # No densification of the scaffold
        if self.scaffold_points is not None:
            selected_pts_mask[:self.scaffold_points] = False
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic)

    def densify_and_prune(self, max_grad, min_opacity, extent, prune_big_points):
        grads = self.xyz_gradient_accum 
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_self_opacity < min_opacity).squeeze()
        if self.scaffold_points is not None:
            prune_mask[:self.scaffold_points] = False

        self.prune_points(prune_mask)

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        for obj_model in self.obj_list:
           obj_model.densify_and_prune(0.0002, min_opacity, prune_big_points) #max_grad
        torch.cuda.empty_cache()

    def set_max_radii2D(self, radii, update_filter):
        start, end = self.graph_gaussian_range['background']
        visibility_model = update_filter[(update_filter >= start) & (update_filter <= end)] - start
        if len(visibility_model) > 0:
            max_radii2D_model = radii[start:end+1]
            self.max_radii2D[visibility_model] = torch.max(self.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
        for obj_model in self.visible_objects:
            start, end = self.graph_gaussian_range[obj_model.get_modelname]
            visibility_model = update_filter[(update_filter >= start) & (update_filter <= end)] - start
            if len(visibility_model) == 0:
                continue
            max_radii2D_model = radii[start:end+1]
            obj_model.max_radii2D[visibility_model] = torch.max(obj_model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        #self.xyz_gradient_accum[update_filter] = torch.max(torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[update_filter])
        #self.denom[update_filter] += 1

        viewspace_point_tensor_grad = viewspace_point_tensor.grad
        start, end = self.graph_gaussian_range['background']
        visibility_model = update_filter[(update_filter >= start) & (update_filter <= end)] - start
        if len(visibility_model) > 0:
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end+1]
            self.xyz_gradient_accum[visibility_model] = torch.max(torch.norm(viewspace_point_tensor_grad_model[visibility_model,:2], dim=-1, keepdim=True), self.xyz_gradient_accum[visibility_model])
            self.denom[visibility_model] += 1

        for obj_model in self.visible_objects:
            start, end = self.graph_gaussian_range[obj_model.get_modelname]
            visibility_model = update_filter[(update_filter >= start) & (update_filter <= end)] - start
            if len(visibility_model) == 0:
                continue
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end+1]
            obj_model.xyz_gradient_accum[visibility_model] = torch.max(torch.norm(viewspace_point_tensor_grad_model[visibility_model,:2], dim=-1, keepdim=True), obj_model.xyz_gradient_accum[visibility_model])
            obj_model.denom[visibility_model] += 1

