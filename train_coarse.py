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
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_gsplat
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

def direct_collate(x):
    return x

def training(dataset, opt, pipe, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(1)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    #viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    target = 0
    indices = None

    iteration = first_iter
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

    for param_group in gaussians.optimizer.param_groups:
        if param_group["name"] == "xyz":
            param_group['lr'] = 0.0

    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                #background = torch.rand((3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                iter_start.record()

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True

                render_pkg = render_gsplat(viewpoint_cam, gaussians, background, use_trained_exp=False, absgrad=dataset.use_absgrad, with_depth=False)
                image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda().float()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda().float()
                    Ll1 = l1_loss(image * alpha_mask, gt_image) 
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * alpha_mask, gt_image))
                else:
                    Ll1 = l1_loss(image, gt_image) 
                    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()

                iter_end.record()

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii)

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}", "Peak memory": f"{torch.cuda.max_memory_allocated(device='cuda')}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in saving_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)

                    if iteration == opt.iterations:
                        progress_bar.close()
                        training_generator._get_iterator()._shutdown_workers()
                        return

                    # Optimizer step

                    if iteration < opt.iterations:
                        if gaussians.road_points > 0: # fix road points
                            # fix xyz, rotation and scaling for sky and road(only optimize color and opacity)
                            gaussians._xyz.grad[:gaussians.road_points, :] = 0
                            gaussians._rotation.grad[:gaussians.road_points, :] = 0
                            gaussians._scaling.grad[:gaussians.road_points, :] = 0
                            #gaussians._features_dc.grad[:gaussians.road_points, :, :] = 0
                            #gaussians._features_rest.grad[:gaussians.road_points, :, :] = 0
                            #gaussians._opacity.grad[:gaussians.road_points, :] = 0
                            
                        fixedpoints = gaussians.road_points + gaussians.skybox_points
                        gaussians._scaling.grad[:fixedpoints,:] = 0
                        relevant = (gaussians._opacity.grad != 0).nonzero()
                        gaussians.optimizer.step(relevant)
                        gaussians.optimizer.zero_grad(set_to_none = True)

                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    with torch.no_grad():
                        vals, _ = gaussians.get_scaling.max(dim=1)
                        violators = vals > scene.cameras_extent * 0.1
                        fixedpoints = gaussians.road_points + gaussians.skybox_points
                        violators[:fixedpoints] = False
                        gaussians._scaling[violators] = gaussians.scaling_inverse_activation(gaussians.get_scaling[violators] * 0.8)


                    iteration += 1


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok = True)
    # training with road model if exists
    roadpoints_3dgs_file = os.path.join(args.model_path, "../road_model/point_cloud/iteration_30000/point_cloud.ply")
    if os.path.exists(roadpoints_3dgs_file):
        args.roadpoints_3dgs_file = roadpoints_3dgs_file
        print("training with road model: ", args.roadpoints_3dgs_file)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
