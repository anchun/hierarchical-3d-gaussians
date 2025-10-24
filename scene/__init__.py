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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import readColmapSceneInfo, fetchPly
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON, CameraDataset
from utils.system_utils import mkdir_p
import numpy as np
from scipy.spatial.transform import Rotation

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, load_filename="point_cloud.ply", shuffle=True, create_from_hier=False,
                        generate_novel_views = False, novel_pos_z = [], novel_rot_z = [], roadpoints_file = None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration <= 0:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.alpha_masks, args.depths, args.eval, args.train_test_exp, 
                                                          use_npy_depth = args.use_npy_depth, eval_camera_name = args.eval_camera_name,
                                                          masks2 = args.road_masks)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)
        novel_cam_infos = []
        if generate_novel_views:
            novel_cam_infos = self.generate_novel_camera_infos(scene_info.train_cameras, novel_pos_z, novel_rot_z)
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            random.shuffle(novel_cam_infos)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Making Training Dataset")
        self.train_cameras = CameraDataset(scene_info.train_cameras, scene_info.point_cloud, args, 1, False)

        print("Making Test Dataset")
        self.test_cameras = CameraDataset(scene_info.test_cameras, scene_info.point_cloud, args, 1, True)
        
        print("Making Novel Dataset")
        self.novel_view_cameras = CameraDataset(novel_cam_infos, scene_info.point_cloud, args, 1, False, is_novel_view=True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           load_filename), scene_info.train_cameras)
        elif args.pretrained:
            self.gaussians.create_from_pt(args.pretrained, self.cameras_extent)
        elif create_from_hier:
            self.gaussians.create_from_hier(args.hierarchy, self.cameras_extent, args.scaffold_file)
        elif roadpoints_file is not None:
            pcd = fetchPly(roadpoints_file)
            self.gaussians.create_from_roadpoints(pcd, 
                                                 scene_info.train_cameras,
                                                 self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, 
                                           scene_info.train_cameras,
                                           self.cameras_extent, 
                                           args.skybox_num,
                                           args.scaffold_file,
                                           args.bounds_file,
                                           args.skybox_locked)


    def save(self, iteration, ply_only=False, filename='point_cloud.ply'):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        mkdir_p(point_cloud_path)
        if self.gaussians.nodes is not None:
            self.gaussians.save_hier()
        else:
            self.gaussians.save_ply(os.path.join(point_cloud_path, filename))
        
        if not ply_only:       
            with open(os.path.join(point_cloud_path, "pc_info.txt"), "w") as f:
                f.write(str(self.gaussians.skybox_points))

            exposure_dict = {
                image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in self.gaussians.exposure_mapping
            }

            with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self):
        return self.train_cameras
    
    def getNovelViewCameras(self):
        return self.novel_view_cameras

    def getTestCameras(self):
        return self.test_cameras
    
    def generate_novel_camera_infos(self, cam_infos, novel_pos_z, novel_rot_z):
        print(f"Generating novel views with pos_z: {novel_pos_z}, rot_z: {novel_rot_z}")
        camera_ids = list(set(cam.uid for cam in cam_infos))
        assert(len(camera_ids)== len(novel_pos_z) and len(camera_ids) == len(novel_rot_z))
        rotations_delta = [Rotation.from_euler('z', rot_z, degrees=True).as_matrix() for rot_z in novel_rot_z]
        positions_delta = [np.array([0, 0, pos_z]) for pos_z in novel_pos_z]
        mid_index = (len(camera_ids) - 1) // 2
        opencv2camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        cam_infos_sorted = sorted(cam_infos, key = lambda x : os.path.basename(x.image_name) + str(x.uid))
        cam_infos_novel = []
        for i in range(0, len(cam_infos_sorted), len(camera_ids)):
            camera_info_mid = cam_infos_sorted[i + mid_index]
            w2c = np.eye(4)
            w2c[:3, :3] = camera_info_mid.R.transpose()
            w2c[:3, 3] = camera_info_mid.T
            c2w = np.linalg.inv(opencv2camera @ w2c)
            for camera_id in camera_ids:
                index = i + camera_id
                camera_info_current = cam_infos_sorted[index]
                current_w2c = np.eye(4)
                current_w2c[:3, :3] = camera_info_current.R.transpose()
                current_w2c[:3, 3] = camera_info_current.T
                current_c2w = np.linalg.inv(opencv2camera @ current_w2c)
                current_c2w[:3, :3] = rotations_delta[camera_id] @ c2w[:3, :3] # override rotation
                current_c2w[:3, 3] += positions_delta[camera_id] # add translation_z
                #print(f"[{camera_id},{camera_info_current.image_name}]position: {current_c2w[:3, 3]}, rotation: {Rotation.from_matrix(current_c2w[:3, :3]).as_euler('zyx', degrees=True)}")
                current_w2c = np.linalg.inv(current_c2w @ opencv2camera)
                camera_info_novel = camera_info_current._replace(
                                        R = current_w2c[:3, :3].transpose(), 
                                        T = current_w2c[:3, 3], 
                                        width = camera_info_current.width // 8 * 8, 
                                        height = camera_info_current.height // 8 * 8, 
                                        primx = camera_info_current.width // 8 * 4,
                                        primy = camera_info_current.height // 8 * 4,
                                        image_path = None, 
                                        mask_path = None,
                                        depth_path = None,
                                        depth_npy_path = None,
                                        depth_params = dict(),
                                        ref_image_path = camera_info_current.image_path)
                cam_infos_novel.append(camera_info_novel)
        return cam_infos_novel

