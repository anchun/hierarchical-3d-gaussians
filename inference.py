import numpy as np
import torch
import os
import json
from tqdm import tqdm
import time
from argparse import ArgumentParser
import sys
from scene import Scene, GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from arguments import ModelParams, PipelineParams, OptimizationParams
from visualizers.scene_visualizer import SceneVisualizer
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--load_iteration', type=int, default=-1)
    args = parser.parse_args(sys.argv[1:])
    args.source_path = ''
    args.model_path = os.path.join(args.input_dir, 'output', 'trained_chunks', '0_0')
    args.resolution = 1
    args.train_test_exp = False

    op = OptimizationParams(parser).extract(args)
    model_params = ModelParams(parser).extract(args)
    model_params.sh_degree = 3
    pipeline_params = PipelineParams(parser).extract(args)
    pipeline_params.debug = False
    pipeline_params.compute_cov3D_python = False
    pipeline_params.convert_SHs_python = False

    print("Rendering " + args.model_path)

    with torch.no_grad():
        model_path = os.path.join(args.input_dir, 'output', 'trained_chunks', '0_0')
        if args.load_iteration == -1:
            args.load_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
        chkp_path = os.path.join(point_cloud_path, 'point_cloud', 'addition_weights.pth')
        print(f'Loading model, iter: {args.load_iteration}')
        state_dict = torch.load(chkp_path)
        saved_ply_folder = os.path.join(model_path, "point_cloud", "iteration_" + str(args.load_iteration))
        scene_info = sceneLoadTypeCallbacks["NOTR"](args.input_dir, os.path.join(args.input_dir, "camera_calibration", "aligned"),
                                                    'camera_calibration/rectifie/images', '',
                                                    '', None, None, None,
                                                    None, True)
        gaussians = GaussianModel(model_params.sh_degree, scene_info, scene_info.scene_meta,
                                  num_camera_poses=len(scene_info.train_cameras),
                                  use_camera_pose_correction=False,
                                  num_classes=0, state_dict=state_dict, saved_ply_folder=saved_ply_folder)
        scene = Scene(model_params, scene_info, gaussians)
        visualizer = SceneVisualizer(args.output_dir)

        cameras = scene.getTrainCameras()
        cameras = [c for c in cameras if c.metadata['cam'] == 0] # cam_0
        cameras = list(sorted(cameras, key=lambda x: x.metadata['frame_id']))

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        visualizer.make_groups({'rgb_gt', 'rgb_inferenced', 'remove_some_vehicles', 'remove_some_and_replace_one'})
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering...")):
            rgb_gt = (camera.original_image[:3].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gaussians.set_visible_dynamic_object_names('all')
            gaussians.replace_inference_models({})
            render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            rgb_inferenced = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            gaussians.set_visible_dynamic_object_names(['obj_010', 'obj_112', 'obj_106', 'obj_030'])
            render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            remove_some_vehicles = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            gaussians.replace_inference_models({'obj_010': 'obj_006'})
            render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            remove_some_and_replace_one = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            visualizer.append_one_frame({
                'rgb_gt': rgb_gt,
                'rgb_inferenced': rgb_inferenced,
                'remove_some_vehicles': remove_some_vehicles,
                'remove_some_and_replace_one': remove_some_and_replace_one,
            })

        output_path = os.path.join(args.output_dir, 'combined_video.mp4')
        os.makedirs(args.output_dir, exist_ok=True)
        sorted_names = ['rgb_gt', 'rgb_inferenced', 'remove_some_vehicles', 'remove_some_and_replace_one']
        layout = {
            "rgb_gt": (0, 0),
            "rgb_inferenced": (0, 1),
            "remove_some_vehicles": (1, 0),
            "remove_some_and_replace_one": (1, 1)
        }
        visualizer.merge_rendered_groups_as_one_mp4(output_path, sorted_names, layout)

