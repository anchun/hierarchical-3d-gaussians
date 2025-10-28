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
import random
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from gaussian_renderer import render_gsplat
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from difix.pipeline_difix import DifixPipeline
import torch.nn.functional as F
import uuid
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

def direct_collate(x):
    return x



def mix_dataloader_sampler(loader1, loader2, p1_schedule, total_iterations):
    it1, it2 = iter(loader1), iter(loader2)
    local_iteration = 0
    while local_iteration < total_iterations:
        p1 = p1_schedule(local_iteration)

        if random.random() < p1:
            try:
                batch = next(it1)
            except StopIteration:
                it1 = iter(loader1)
                batch = next(it1)
        else:
            try:
                batch = next(it2)
            except StopIteration:
                it2 = iter(loader2)
                batch = next(it2)

        yield batch
        local_iteration += 1

def training(dataset, opt, pipe, args):
    first_iter = 0
    checkpoint = args.start_checkpoint
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, generate_novel_views = args.generate_novel_views, novel_pos_z = args.novel_pos_z, novel_rot_z = args.novel_rot_z)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    psnr_val_for_log = 0.0
    ssim_val_for_log = 0.0
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, batch_size = 1, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
    fix_generator = DataLoader(scene.getNovelViewCameras(), num_workers = 8, batch_size = 1, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)
    # Diffusion fixer
    if args.generate_novel_views:
        difix = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
        difix.set_progress_bar_config(disable=True)
        difix.to("cuda")
    else:
        difix = None
    
    torch.cuda.set_per_process_memory_fraction(0.9, device=None)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for viewpoint_batch in mix_dataloader_sampler(training_generator, fix_generator, lambda it: 1.0 if (it < opt.fix_from_iter or not args.generate_novel_views) else 0.7, opt.iterations):
            for viewpoint_cam in viewpoint_batch:

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == args.debug_from:
                    pipe.debug = True
                render_pkg = render_gsplat(viewpoint_cam, gaussians, background, use_trained_exp=True, absgrad=dataset.use_absgrad, with_inv_depth=False)
                image, depths, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                if not viewpoint_cam.is_novel_view:
                    gt_image = viewpoint_cam.original_image.cuda()
                    if viewpoint_cam.alpha_mask is not None:
                        alpha_mask = viewpoint_cam.alpha_mask.cuda()
                        image *= alpha_mask
                else:
                    gt_image = difix(prompt="remove degradation", 
                                     image=image, 
                                     ref_image=viewpoint_cam.original_image.cuda(),
                                     num_inference_steps=1, timesteps=[199], 
                                     guidance_scale=0.0, 
                                     output_type='pt').images[0]
                    #gt_image = gt_image.resize(gt_image.size, Image.LANCZOS)
                
                Ll1 = l1_loss(image, gt_image)
                Lssim = (1.0 - ssim(image, gt_image))
                psnr_val = psnr(image, gt_image).mean().double()
                ssim_val = (1.0 - Lssim).mean().double()
                if viewpoint_cam.is_novel_view:
                    print("[", iteration, "] novel view", " PSNR: ", psnr_val.item(), " SSIM: ", ssim_val.item())

                photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim 
                loss = photo_loss.clone()
                Ll1depth = 0.0
                if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                    depth_mask = viewpoint_cam.depth_mask.cuda() if viewpoint_cam.depth_mask is not None else None
                    if dataset.use_npy_depth and viewpoint_cam.invdepthmap_npy is not None:
                        invdepthmap_npy = viewpoint_cam.invdepthmap_npy.cuda()
                        depth_gt = invdepthmap_npy[:, 2]
                        points = torch.stack(
                            [
                                invdepthmap_npy[:, 0] / (viewpoint_cam.image_width - 1) * 2 - 1,
                                invdepthmap_npy[:, 1] / (viewpoint_cam.image_height - 1) * 2 - 1,
                            ],
                            dim=-1,
                        )  # normalize to [-1, 1] [M, 2]
                        grid = points.unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]
                        if depth_mask is not None:
                            depths = depths * depth_mask
                        depths = F.grid_sample(
                            depths.unsqueeze(0), grid, align_corners=True
                        )  # [1, 1, M, 1]
                        depths = depths.squeeze()
                        invDepths = 1.0 / depths[depths > 0.0]
                        depth_gt = depth_gt[depths > 0.0]
                        depth_error = torch.abs(invDepths - depth_gt)
                        depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)
                        L1_depth_loss_npy = opt.depth_loss_weight * depth_error.mean()
                        loss += L1_depth_loss_npy
                        Ll1depth += L1_depth_loss_npy.item()
                    elif viewpoint_cam.invdepthmap is not None:
                        invDepths = 1.0 / depths.clamp(min=1e-10)
                        mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                        depth_error = torch.abs((invDepths  - mono_invdepth) * depth_mask)
                        L1_depth_loss = depth_l1_weight(iteration) * depth_error.mean() 
                        loss += L1_depth_loss
                        Ll1depth += L1_depth_loss.item()

                loss.backward()
                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.1 * photo_loss.item() + 0.9 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.1 * Ll1depth + 0.9 * ema_Ll1depth_for_log
                    psnr_val_for_log = 0.1 * psnr_val + 0.9 * psnr_val_for_log
                    ssim_val_for_log = 0.1 * ssim_val + 0.9 * ssim_val_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "PSNR": f"{psnr_val_for_log:.{5}f}", "SSIM": f"{ssim_val_for_log:.{5}f}" , "Size": f"{gaussians._xyz.size(0)}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in args.save_iterations):
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration % opt.opacity_reset_interval == 0:
                        print()
                    if iteration == opt.iterations:
                        progress_bar.close()
                        return

                    # Densification
                    if iteration < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], use_absgrad = dataset.use_absgrad)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            prune_big_points = iteration > opt.opacity_reset_interval
                            gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_absgrad_threshold, opt.max_gaussian_num, opt.min_opacity, scene.cameras_extent, prune_big_points, use_absgrad=dataset.use_absgrad)
                        
                        if iteration % opt.opacity_reset_interval == 0:
                            #print("-----------------RESET OPACITY!-------------")
                            gaussians.reset_opacity()

                    # Optimizer step
                    if iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                        if gaussians._xyz.grad != None and gaussians.skybox_locked:
                            # fixed for road and sky points
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0

                        if gaussians._opacity.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if(relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            else:
                                gaussians.optimizer.step(relevant)
                                print("No grads!")
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if not args.skip_scale_big_gauss:
                        with torch.no_grad():
                            vals, _ = gaussians.get_scaling.max(dim=1)
                            violators = vals > scene.cameras_extent * 0.02
                            if gaussians.scaffold_points is not None:
                                violators[:gaussians.scaffold_points] = False
                            gaussians._scaling[violators] = gaussians.scaling_inverse_activation(gaussians.get_scaling[violators] * 0.8)
                        
                    if (iteration in args.checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument('--generate_novel_views', action='store_true', default=False)
    parser.add_argument("--novel_pos_z", nargs="+", type=int, default=[0, 0, 1, 0, 0])
    parser.add_argument("--novel_rot_z", nargs="+", type=int, default=[150, 90, 0, -90, -150])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Iterations: ", args.iterations, "Densify iterations: ", args.densify_until_iter, "degree SH: ", args.sh_degree)
    
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok = True)
    # training with road model if exists
    roadpoints_3dgs_file = os.path.join(args.model_path, "../../road_model/point_cloud/iteration_30000/point_cloud.ply")
    if os.path.exists(roadpoints_3dgs_file):
        args.roadpoints_3dgs_file = roadpoints_3dgs_file
        print("training with road model: ", args.roadpoints_3dgs_file)

    if args.eval and args.exposure_lr_init > 0 and not args.train_test_exp: 
        print("Reconstructing for evaluation (--eval) with exposure optimization on the train set but not for the test set.")
        print("This will lead to high error when computing metrics. To optimize exposure on the left half of the test images, use --train_test_exp")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #if not args.disable_viewer:
        #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
