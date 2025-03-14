import torch
import torch.nn as nn
import numpy as np
import os
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, quaternion_to_matrix, quaternion_raw_multiply, matrix_to_quaternion
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from preprocess.read_write_model import rotmat2qvec
from scene.cameras import Camera
import time
from scene.OurAdam import Adam


def IDFT(time, dim):
    if isinstance(time, float):
        time = torch.tensor(time)
        t = time.view(-1, 1).float()
        idft = torch.zeros(t.shape[0], dim)
        indices = torch.arange(dim)
        even_indices = indices[::2]
        odd_indices = indices[1::2]
        idft[:, even_indices] = torch.cos(torch.pi * t * even_indices)
        idft[:, odd_indices] = torch.sin(torch.pi * t * (odd_indices + 1))
        return idft


class GaussianModelActor():
    def __init__(
        self, 
        obj_meta=None,
        num_frames=None,
        max_sh_degree=1,
        num_classes=0,
    ):
        #cfg_model = cfg.model.gaussian

        # semantic
        self.num_classes = num_classes
        self.semantic_mode = 'logits'

        # spherical harmonics
        self.active_sh_degree = 1

        # original gaussian initialization
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
        self.setup_functions()

        # parse obj_meta
        self.obj_meta = obj_meta
        
        self.obj_class = obj_meta['class']
        self.obj_class_label = obj_meta['class_label']
        self.deformable = obj_meta['deformable']
        self.start_frame = obj_meta['start_frame']
        self.start_timestamp = obj_meta['start_timestamp']
        self.end_frame = obj_meta['end_frame']
        self.end_timestamp = obj_meta['end_timestamp']
        self.object_id = obj_meta['object_id']
        self.num_frames = num_frames

        # fourier spherical harmonics
        self.fourier_dim = 1 #cfg.model.gaussian.get('fourier_dim', 1) TODO
        self.fourier_scale = 1 #cfg.model.gaussian.get('fourier_scale', 1.) TODO
        
        # bbox
        length, width, height = obj_meta['length'], obj_meta['width'], obj_meta['height']
        self.bbox = np.array([length, width, height]).astype(np.float32)
        xyz = torch.tensor(self.bbox).float().cuda()
        self.min_xyz, self.max_xyz =  -xyz/2., xyz/2.  
        
        #extent = max(length*1.5/cfg.data.box_scale, width*1.5/cfg.data.box_scale, height) / 2.
        box_scale=1.5
        extent = max(length*1.5/box_scale, width*1.5/box_scale, height) / 2.
        self.extent = torch.tensor([extent]).float().cuda()   

        num_classes = 0 # 1 if cfg.data.get('use_semantic', False) else 0
        self.num_classes_global = 1 # cfg.data.num_classes if cfg.data.get('use_semantic', False) else 0 TODO
        #super().__init__(model_name=model_name, num_classes=num_classes)
        
        self.flip_prob = 0 # cfg.model.gaussian.get('flip_prob', 0.) if not self.deformable else 0.
        self.flip_axis = 1

        self.spatial_lr_scale = extent

        self.max_sh_degree = max_sh_degree # TODO 什么用途？

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

    def get_extent(self):
        max_scaling = torch.max(self.get_scaling, dim=1).values

        extent_lower_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=False).values[-1] / self.percent_dense
        extent_upper_bound = torch.topk(max_scaling, int(self.get_xyz.shape[0] * 0.1), largest=True).values[-1] / self.percent_dense
        
        extent = torch.clamp(self.extent, min=extent_lower_bound, max=extent_upper_bound)        
        print(f'extent: {extent.item()}, extent bound: [{extent_lower_bound}, {extent_upper_bound}]')

        return extent

    @property
    def get_modelname(self):
        return f'obj_{self.object_id:03d}'

    @property
    def get_semantic(self):
        # semantic = torch.zeros((self.get_xyz.shape[0], self.num_classes_global)).float().cuda()
        # if self.semantic_mode == 'logits':
        #     semantic[:, self.obj_class_label] = self._semantic[:, 0] # ubounded semantic
        # elif self.semantic_mode == 'probabilities':
        #     semantic[:, self.obj_class_label] = torch.nn.functional.sigmoid(self._semantic[:, 0]) # 0 ~ 1
        #
        # return semantic
        if True:  # self.semantic_mode == 'logits':
            return self._semantic
        else: # 'probabilities':
            return torch.nn.functional.softmax(self._semantic, dim=1)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_normals(self, camera: Camera):
        scales, rotations = self.get_scaling, self.get_rotation
        rotations_mat = quaternion_to_matrix(rotations)
        min_scales = torch.argmin(scales, dim=-1)
        indices = torch.arange(min_scales.shape[0])
        normals = rotations_mat[indices, :, min_scales]

        # points from gaussian to camera
        dir_pp = (self.get_xyz - camera.camera_center.repeat(self._xyz.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)  # (N, 3)
        dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True)  # (N, 1)
        normals = torch.where(dotprod >= 0, normals, -normals)

        return normals

    def scale_flatten_loss(self):
        scales = self.get_scaling
        sorted_scales = torch.sort(scales, dim=1, descending=False).values
        s1, s2, s3 = sorted_scales[:, 0], sorted_scales[:, 1], sorted_scales[:, 2]
        s1 = torch.clamp(s1, 0, 30)
        s2 = torch.clamp(s2, 1e-5, 30)
        s3 = torch.clamp(s3, 1e-5, 30)
        scale_flatten_loss = torch.abs(s1).mean()
        scale_flatten_loss += torch.abs(s2 / s3 + s3 / s2 - 2.).mean()
        return scale_flatten_loss

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def get_features_fourier(self, frame_id):
        normalized_frame = (frame_id - self.start_frame) / (self.end_frame - self.start_frame)
        time = self.fourier_scale * normalized_frame

        idft_base = IDFT(time, self.fourier_dim)[0].cuda()
        features_dc = self._features_dc # [N, C, 3]
        features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True) # [N, 1, 3]
        features_rest = self._features_rest # [N, sh, 3]
        features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        return features
           
    def create_from_pcd(self, point_clouds):
        if self.get_modelname in point_clouds.keys():
            pcd = point_clouds[self.get_modelname]
            pointcloud_xyz = np.asarray(pcd.points)
            if pointcloud_xyz.shape[0] < 2000:
                self.random_initialization = True
            else:
                self.random_initialization = False
        else:
            self.random_initialization = True

        if self.random_initialization is True:
            points_dim = 20
            print(f'Creating random pointcloud for {self.get_modelname}')
            points_x, points_y, points_z = np.meshgrid(
                np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim),
            )
            
            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)
            points_z = points_z.reshape(-1)

            bbox_xyz_scale = self.bbox / 2.
            pointcloud_xyz = np.stack([points_x, points_y, points_z], axis=-1)
            pointcloud_xyz = pointcloud_xyz * bbox_xyz_scale            
            pointcloud_rgb = np.random.rand(*pointcloud_xyz.shape).astype(np.float32)  
        elif not self.deformable and self.flip_prob > 0.:
            # pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            num_pointcloud_1 = (pointcloud_xyz[:, self.flip_axis] > 0).sum()
            num_pointcloud_2 = (pointcloud_xyz[:, self.flip_axis] < 0).sum()
            if num_pointcloud_1 >= num_pointcloud_2:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] > 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] > 0]
            else:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] < 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] < 0]
            pointcloud_xyz_flip = pointcloud_xyz_part.copy()
            pointcloud_xyz_flip[:, self.flip_axis] *= -1
            pointcloud_rgb_flip = pointcloud_rgb_part.copy()
            pointcloud_xyz = np.concatenate([pointcloud_xyz, pointcloud_xyz_flip], axis=0)
            pointcloud_rgb = np.concatenate([pointcloud_rgb, pointcloud_rgb_flip], axis=0)
        else:
            # pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)

        fused_point_cloud = torch.tensor(np.asarray(pointcloud_xyz)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pointcloud_rgb)).float().cuda())
        print(f"Number of points at initialisation for {self.get_modelname}: ", fused_point_cloud.shape[0])

        # features = torch.zeros((fused_color.shape[0], 3, 
        #                         (self.max_sh_degree + 1) ** 2 * self.fourier_dim)).float().cuda()
        # features[:, :3, 0] = fused_color
        features_dc = torch.zeros((fused_color.shape[0], 3, self.fourier_dim)).float().cuda()
        features_rest = torch.zeros(fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1).float().cuda()
        features_dc[:, :3, 0] = fused_color

        #print(f"Number of points at initialisation for {self.get_modelname}: ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pointcloud_xyz)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1))).float().cuda()
        semantics = torch.zeros((fused_point_cloud.shape[0], self.num_classes)).float().cuda()
        self._xyz = nn.Parameter(fused_point_cloud.cuda().requires_grad_(True))
        
        # self._features_dc = nn.Parameter(features[:, :, :self.fourier_dim].transpose(1, 2).contiguous().requires_grad_(True))
        # self._features_rest = nn.Parameter(features[:, :, self.fourier_dim:].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.cuda().transpose(1, 2).contiguous().requires_grad_(True))
        
        self._scaling = nn.Parameter(scales.cuda().requires_grad_(True))
        self._rotation = nn.Parameter(rots.cuda().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.cuda().requires_grad_(True))
        self._semantic = nn.Parameter(semantics.cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def make_ply(self):
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

        return elements

    def load_ply(self, path=None, input_ply=None):
        if path is None:
            plydata = input_ply
        else:
            plydata = PlyData.read(path)
            plydata = plydata.elements[0]

        xyz = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])), axis=1)
        opacities = np.asarray(plydata["opacity"])[..., np.newaxis]

        base_f_names = [p.name for p in plydata.properties if p.name.startswith("f_dc_")]
        base_f_names = sorted(base_f_names, key=lambda x: int(x.split('_')[-1]))
        extra_f_names = [p.name for p in plydata.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        features_dc = np.zeros((xyz.shape[0], len(base_f_names)))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(base_f_names):
            features_dc[:, idx] = np.asarray(plydata[attr_name])
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata[attr_name])
        features_dc = features_dc.reshape(features_dc.shape[0], 3, -1)
        features_extra = features_extra.reshape(features_extra.shape[0], 3, -1)

        scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name])

        rot_names = [p.name for p in plydata.properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name])

        semantic_names = [p.name for p in plydata.properties if p.name.startswith("semantic_")]
        semantic_names = sorted(semantic_names, key=lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((xyz.shape[0], len(semantic_names)))
        if self.num_classes > 0:
            for idx, attr_name in enumerate(semantic_names):
                semantic[:, idx] = np.asarray(plydata[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def restore(self, state_dict):
        self._xyz = state_dict['xyz']
        self._features_dc = state_dict['feature_dc']
        self._features_rest = state_dict['feature_rest']
        self._scaling = state_dict['scaling']
        self._rotation = state_dict['rotation']
        self._opacity = state_dict['opacity']
        self._semantic = state_dict['semantic']

    def restore_training_status(self, state_dict):
        if 'spatial_lr_scale' in state_dict:
            self.spatial_lr_scale = state_dict['spatial_lr_scale']
        if 'denom' in state_dict:
            self.denom = state_dict['denom']
        if 'max_radii2D' in state_dict:
            self.max_radii2D = state_dict['max_radii2D']
        if 'xyz_gradient_accum' in state_dict:
            self.xyz_gradient_accum = state_dict['xyz_gradient_accum']
        if 'active_sh_degree' in state_dict:
            self.active_sh_degree = state_dict['active_sh_degree']
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def save_state_dict(self):
        state_dict = {
            'xyz': self._xyz,
            'feature_dc': self._features_dc,
            'feature_rest': self._features_rest,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
            'semantic': self._semantic,
            'spatial_lr_scale': self.spatial_lr_scale,
            'denom': self.denom,
            'max_radii2D': self.max_radii2D,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'active_sh_degree': self.active_sh_degree,
            'optimizer': self.optimizer.state_dict(),
        }
        return state_dict

    def training_setup(self, training_args):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.active_sh_degree = 0

        # TODO lr of objs must not the same as background
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': training_args.semantic_lr, "name": "semantic"},
        ]
        
        self.percent_dense = 0.01 # training_args.percent_dense
        self.percent_big_ws = 0.1 # training_args.percent_big_ws
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # self.optimizer = Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
        self.densify_and_prune_list = ['xyz, f_dc, f_rest, opacity, scaling, rotation, semantic']
        self.scalar_dict = dict()
        self.tensor_dict = dict()

    def update_optimizer(self):
        # if self._opacity.grad != None:
        #    relevant = (self._opacity.grad.flatten() != 0).nonzero()
        #    relevant = relevant.flatten().long()
        #    self.optimizer.step(relevant)
        #    self.optimizer.zero_grad(set_to_none = True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def prune_optimizer(self, mask, prune_list=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] not in prune_list:
                continue

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

    def cat_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        name_list = tensors_dict.keys()
        for group in self.optimizer.param_groups:
            if group['name'] not in name_list:
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
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


    def reset_optimizer(self, tensors_dict):
        optimizable_tensors = {}
    
        name_list = tensors_dict.keys()
        for group in self.optimizer.param_groups:
            if group['name'] in name_list:
                reset_tensor = tensors_dict[group['name']]

                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(reset_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(reset_tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(reset_tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
    
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densify_and_prune(self, max_grad, min_opacity, prune_big_points):
        if not (self.random_initialization or self.deformable):
            # max_grad = cfg.optim.get('densify_grad_threshold_obj', max_grad)
            if False: #cfg.optim.get('densify_grad_abs_obj', False):
                grads = self.xyz_gradient_accum[:, 1:2] / self.denom
            else:
                grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        else:
            grads = self.xyz_gradient_accum[:, 0:1] / self.denom
        
        grads[grads.isnan()] = 0.0

        # Clone and Split
        # extent = self.get_extent()
        extent = self.extent
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune points below opacity
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        if prune_big_points:
            # Prune big points in world space
            extent = self.extent
            big_points_ws = self.get_scaling.max(dim=1).values > extent * self.percent_big_ws
            
            # Prune points outside the tracking box
            repeat_num = 2
            stds = self.get_scaling
            stds = stds[:, None, :].expand(-1, repeat_num, -1) # [N, M, 1] 
            means = torch.zeros_like(self.get_xyz)
            means = means[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
            samples = torch.normal(mean=means, std=stds) # [N, M, 3]
            rots = quaternion_to_matrix(self.get_rotation) # [N, 3, 3]
            rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1) # [N, M, 3, 3]
            origins = self.get_xyz[:, None, :].expand(-1, repeat_num, -1) # [N, M, 3]
                        
            samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins # [N, M, 3]                    
            num_gaussians = self.get_xyz.shape[0]
            if num_gaussians > 0:
                points_inside_box = torch.logical_and(
                    torch.all((samples_xyz >= self.min_xyz).view(num_gaussians, -1), dim=-1),
                    torch.all((samples_xyz <= self.max_xyz).view(num_gaussians, -1), dim=-1),
                )
                points_outside_box = torch.logical_not(points_inside_box)           
            
                prune_mask = torch.logical_or(prune_mask, big_points_ws)
                prune_mask = torch.logical_or(prune_mask, points_outside_box)
            
        # print(f'==== densify_and_prune {self.get_modelname}: number of points to prune: {prune_mask.sum()}')
        self.prune_points(prune_mask)
        
        # Reset
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 2), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        torch.cuda.empty_cache()
        
        return self.scalar_dict, self.tensor_dict

    def densification_postfix(self, tensors_dict):
        optimizable_tensors = self.cat_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]

        cat_points_num = self.get_xyz.shape[0] - self.xyz_gradient_accum.shape[0]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros(cat_points_num, 2).cuda()], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros(cat_points_num, 1).cuda()], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(cat_points_num).cuda()], dim=0)

        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_extent = torch.zeros((n_init_points), device="cuda")
        padded_extent[:grads.shape[0]] = scene_extent

        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling,
        #                                                 dim=1).values > self.percent_dense * padded_extent)
        selected_pts_mask = torch.where(padded_grad * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        self.scalar_dict['points_split'] = selected_pts_mask.sum().item()
        # print(f'==== Split {self.get_modelname}: number of points to split: {selected_pts_mask.sum()}')

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_semantic = self._semantic[selected_pts_mask].repeat(N, 1)

        self.densification_postfix({
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "semantic": new_semantic,
        })

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # print(f'==== densify_and_split {self.get_modelname}: number of points to prune: {prune_filter.sum()}')
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):

        # Extract points that satisfy the gradient condition
        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling,
        #                                                 dim=1).values <= self.percent_dense * scene_extent)
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * self.max_radii2D * torch.pow(self.get_opacity.flatten(), 1/5.0) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, self.get_opacity.flatten() > 0.15)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        self.scalar_dict['points_clone'] = selected_pts_mask.sum().item()
        #print(f'Number of points to clone: {selected_pts_mask.sum()}')
        # print(f'==== Clone {self.get_modelname}: number of points to clone: {selected_pts_mask.sum()}')

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic = self._semantic[selected_pts_mask]

        self.densification_postfix({
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacity,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "semantic": new_semantic,
        })

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor.grad[update_filter, :2],
    #                                                               dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1
    #
    # def add_densification_stats_grad(self, viewspace_point_tensor_grad, update_filter):
    #     self.xyz_gradient_accum[update_filter, 0:1] += torch.norm(viewspace_point_tensor_grad[update_filter, :2],
    #                                                               dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1
    #
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        d = {'opacity': opacities_new}
        optimizable_tensors = self.reset_optimizer(d)
        self._opacity = optimizable_tensors["opacity"]

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self.prune_optimizer(valid_points_mask,
            prune_list = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation', 'semantic'])

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

    def set_max_radii(self, visibility_obj, max_radii2D):
        self.max_radii2D[visibility_obj] = torch.max(self.max_radii2D[visibility_obj], max_radii2D[visibility_obj])
    
    def box_reg_loss(self):
        scaling_max = self.get_scaling.max(dim=1).values
        scaling_max = torch.where(scaling_max > self.extent * self.percent_dense, scaling_max, 0.)
        reg_loss = (scaling_max / self.extent).mean()
        
        return reg_loss
        
        
    
    
