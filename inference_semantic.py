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
from PIL import Image


def parse_visible_obj_names(visible_obj_ids):
    if visible_obj_ids in ['all', 'none']:
        return visible_obj_ids
    ids = visible_obj_ids.split(',')
    return ['obj_' + id for id in ids]


def parse_visible_obj_replacements(visible_obj_replacements):
    if visible_obj_replacements == '' or visible_obj_replacements is None:
        return {}
    replacements = {}
    for o1_o2 in visible_obj_replacements.split(','):
        splited = o1_o2.split(':')
        objname1 = 'obj_' + splited[0]
        objname2 = 'obj_' + splited[1]
        replacements[objname1] = objname2
    return replacements


def load_scene(args, model_params):
    model_path = os.path.join(args.input_dir, 'output', 'trained_chunks', '0_0')
    if args.load_iteration == -1:
        args.load_iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    saved_ply_folder = os.path.join(model_path, "point_cloud", "iteration_" + str(args.load_iteration))
    chkp_path = os.path.join(saved_ply_folder, 'addition_weights.pth')
    print(f'Loading model, iter: {args.load_iteration}')
    state_dict = torch.load(chkp_path)
    scene_info = sceneLoadTypeCallbacks["NOTR"](args.input_dir,
                                                os.path.join(args.input_dir, "camera_calibration", "aligned"),
                                                '../rectified/images', '',
                                                '', None, None, None,
                                                None, True)
    gaussians = GaussianModel(model_params, scene_info, scene_info.scene_meta,
                              num_camera_poses=len(scene_info.train_cameras),
                              use_camera_pose_correction=False,
                              num_classes=10, state_dict=state_dict, saved_ply_folder=saved_ply_folder)
    scene = Scene(model_params, scene_info, gaussians)
    return scene, gaussians


def to_semantic_class(logits):
    """
        input: tensor with shape [num_semantic_class, w, h], means semantic logits
        return: numpy array with shape: [1, w, h], means semantic class
    """
    logits = logits.detach().cpu().numpy()
    classes = np.argmax(logits, axis=0).astype(np.uint8)
    # 增加一个维度，使其形状变为 [1, w, h]
    return np.expand_dims(classes, axis=0)


def blend_semantic_on_rgb(rgb_gt, semantic, alpha=0.5):
    """
        根据语义类修改rgb图像，前景（类别0）不变，其他9类分别用不同颜色半透明叠加上去。
    Args:
        rgb_gt (np.ndarray): 原RGB图像，形状为(h, w, 3)，数据类型通常为np.uint8。
        semantic (np.ndarray): 语义分割结果，形状为(h, w, 1)，值为0表示前景，1-9表示其他类别。
        alpha (float): 透明度，取值范围为 [0, 1]，0表示完全透明，1表示完全不透明，默认为0.5。
    Returns:
        np.ndarray: 修改后的RGB图像，形状为(h, w, 3)，数据类型与rgb_gt相同。
    """
    # 为除前景（类别0）外的9个类别定义不同颜色
    class_colors = [
        [0, 255, 0],    # 绿色
        [0, 0, 128],     # 深蓝色
        [0, 0, 255],    # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 紫色
        [0, 255, 255],  # 青色
        [128, 0, 0],    # 深红色
        [0, 128, 0],    # 深绿色
        [255, 0, 0],    # 红色
    ]
    rgb_blended = rgb_gt.astype(np.float32).copy()
    for class_id in range(1, 10):
        class_mask = (semantic == class_id).squeeze(-1)  # 将(h, w, 1)变成(h, w)的布尔mask
        class_rgb = np.array(class_colors[class_id - 1]).reshape(1, 1, 3)
        rgb_blended[class_mask] = (1 - alpha) * rgb_gt[class_mask].astype(np.float32) + alpha * class_rgb
    return np.clip(rgb_blended, 0, 255).astype(rgb_gt.dtype)


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--visible_obj_ids', type=str, default='')
    parser.add_argument('--visible_obj_replacements', type=str, default='')
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
    semantic_output_folder = os.path.join(args.output_dir, 'render_semantics_cam_' + str(0))
    os.makedirs(semantic_output_folder, exist_ok=True)
    with torch.no_grad():
        scene, gaussians = load_scene(args, model_params)
        visualizer = SceneVisualizer(args.output_dir)

        cameras = scene.getTrainCameras()
        cameras = [c for c in cameras if c.metadata['cam'] == 0] # cam_0
        cameras = list(sorted(cameras, key=lambda x: x.metadata['frame_id']))

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        visualizer.make_groups(['rgb_gt', 'rgb_inferenced'
                                   # , 'remove_some_vehicles', 'remove_some_and_replace_one'
                                   , 'rgb_gt_with_semantic', 'rgb_inferenced_with_semantic'
                               ])
        visible_obj_names = parse_visible_obj_names(args.visible_obj_ids)
        replacements = parse_visible_obj_replacements(args.visible_obj_replacements)
        for idx, camera in enumerate(tqdm(cameras, desc="Rendering...")):
            rgb_gt = (camera.original_image[:3].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            gaussians.set_visible_dynamic_object_names('all')
            gaussians.replace_inference_models({})
            render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            rgb_inferenced = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            # gaussians.set_visible_dynamic_object_names(visible_obj_names)
            # render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            # remove_some_vehicles = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            #
            # gaussians.replace_inference_models(replacements)
            # render_result = render(camera, gaussians, pipeline_params, background, indices=None, use_trained_exp=False)
            # remove_some_and_replace_one = (render_result['render'].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            semantic_gt = camera.semantic.detach().cpu().numpy().transpose(1, 2, 0)
            semantic_inferenced = to_semantic_class(render_result['semantic'])
            semantic_inferenced = semantic_inferenced.transpose(1, 2, 0).astype(semantic_gt.dtype)
            semantic_img = Image.fromarray(semantic_inferenced.squeeze(-1).astype(np.uint8)*255, mode='L')
            semantic_img.save(os.path.join(semantic_output_folder, f'{idx:06d}.jpg'))
            rgb_gt_with_semantic = blend_semantic_on_rgb(rgb_gt, semantic_gt)

            rgb_inferenced_with_semantic = blend_semantic_on_rgb(rgb_inferenced, semantic_inferenced)

            visualizer.append_one_frame({
                'rgb_gt': rgb_gt,
                'rgb_inferenced': rgb_inferenced,
                # 'remove_some_vehicles': remove_some_vehicles,
                # 'remove_some_and_replace_one': remove_some_and_replace_one,
                'rgb_gt_with_semantic': rgb_gt_with_semantic,
                'rgb_inferenced_with_semantic': rgb_inferenced_with_semantic,
            })

        output_path = os.path.join(args.output_dir, 'combined_video.mp4')
        os.makedirs(args.output_dir, exist_ok=True)
        layout = {
            "rgb_gt": (0, 0),
            "rgb_inferenced": (0, 1),
            # "remove_some_vehicles": (1, 0),
            # "remove_some_and_replace_one": (1, 1),
            "rgb_gt_with_semantic": (1, 0),
            "rgb_inferenced_with_semantic": (1, 1),
        }
        print('Saving videos...')
        visualizer.merge_rendered_groups_as_one_mp4(output_path, layout)

