import os
import torch
import random
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from gaussian_renderer import render, render_gsplat, render_gsplat2d
import sys
from scene import Scene, GaussianModel
from functools import partial
from utils.general_utils import safe_state
from difix.pipeline_difix import DifixPipeline
import uuid
from tqdm import tqdm
import numpy as np
import open3d as o3d
import torch.nn.functional as F
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

def rendering_mesh(gaussians, training_cameras, gaussian_render, args):
    gaussExtractor = GaussianExtractor(gaussians, gaussian_render)    

    print("export mesh ...")
    mesh_output_dir = os.path.join(args.model_path, 'mesh')
    os.makedirs(mesh_output_dir, exist_ok=True)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(training_cameras)
    # extract the mesh and save
    name = 'road_mesh.ply'
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=0.025, sdf_trunc=0.5, depth_trunc=20.0)
    
    #o3d.io.write_triangle_mesh(os.path.join(mesh_output_dir, name), mesh)
    #print("mesh saved at {}".format(os.path.join(mesh_output_dir, name.replace('.ply', '_raw.ply'))))
    
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
    o3d.io.write_triangle_mesh(os.path.join(mesh_output_dir, name), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_output_dir, name)))
    
def training(dataset, opt, pipe, args):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration = args.load_iteration, load_filename="road_point_cloud.ply", generate_novel_views = args.generate_novel_views, novel_pos_z = args.novel_pos_z, novel_rot_z = args.novel_rot_z, roadpoints_file = args.roadpoints_file)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    gaussian_render = partial(render_gsplat2d if dataset.use_gsplat2d else render_gsplat, bg_color=background, use_trained_exp=True, absgrad=dataset.use_absgrad, with_inv_depth=False)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    psnr_val_for_log = 0.0
    ssim_val_for_log = 0.0
    depth_val_for_log = 0.0
    normal_val_for_log = 0.0
    
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
    first_iter = args.load_iteration if args.load_iteration > 0 else 1
    progress_bar = tqdm(range(first_iter, opt.iterations+1), desc="Training progress")
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for viewpoint_batch in mix_dataloader_sampler(training_generator, fix_generator, lambda it: 1.0 if (it < opt.fix_from_iter or not args.generate_novel_views) else 0.7, opt.iterations):
            for viewpoint_cam in viewpoint_batch:

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()
                viewpoint_cam.invdepthmap_npy = viewpoint_cam.invdepthmap_npy.cuda()

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == args.debug_from:
                    pipe.debug = True
                render_pkg = gaussian_render(viewpoint_cam, gaussians)
                image, depths = render_pkg["render"], render_pkg["depth"]
                if dataset.use_gsplat2d:
                    normals, normals_from_depth = render_pkg["normal"], render_pkg["normals_from_depth"]

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
                depth_gt = viewpoint_cam.invdepthmap_npy[:, 2]
                points = torch.stack(
                    [
                        viewpoint_cam.invdepthmap_npy[:, 0] / (viewpoint_cam.image_width - 1) * 2 - 1,
                        viewpoint_cam.invdepthmap_npy[:, 1] / (viewpoint_cam.image_height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1] [M, 2]
                grid = points.unsqueeze(0).unsqueeze(2)  # [1, M, 1, 2]
                depths *= alpha_mask
                depths = F.grid_sample(
                    depths.unsqueeze(0), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze()
                invDepths = 1.0 / depths[depths > 0.0]
                depth_gt = depth_gt[depths > 0.0]
                depthloss = F.l1_loss(invDepths, depth_gt) * opt.depth_loss_weight
                loss += depthloss
                
                normal_val = 0.0
                if dataset.use_gsplat2d:
                    # add normal loss
                    normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                    normalloss = opt.normal_loss_weight * normal_error.mean()
                    normal_val = normalloss.item()
                    loss += normalloss

                loss.backward()
                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.1 * photo_loss.item() + 0.9 * ema_loss_for_log
                    psnr_val_for_log = 0.1 * psnr_val + 0.9 * psnr_val_for_log
                    ssim_val_for_log = 0.1 * ssim_val + 0.9 * ssim_val_for_log
                    normal_val_for_log = 0.1 * normal_val + 0.9 * normal_val_for_log
                    depth_val_for_log = 0.1 * depthloss.item() + 0.9 * depth_val_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{3}f}", "PSNR": f"{psnr_val_for_log:.{3}f}", "SSIM": f"{ssim_val_for_log:.{3}f}" , "Depth": f"{depth_val_for_log:.{3}f}", "Normal": f"{normal_val_for_log:.{3}f}" , "Size": f"{gaussians._xyz.size(0)}"})
                        progress_bar.update(10)

                    # Log and save
                    if (iteration in args.save_iterations):
                        if iteration == opt.iterations:
                            progress_bar.close()
                            print("clean up big road points...")
                            road_mean_distance = gaussians.gaussian_road_mean_distance
                            if args.use_gsplat2d:
                                mean_scale = road_mean_distance ** 2
                                large_gaussians = gaussians.get_scaling[:, :2].prod(dim=1) / mean_scale > (args.max_valid_scale ** 2)
                            else:
                                mean_scale = road_mean_distance ** 3
                                large_gaussians = gaussians.get_scaling.prod(dim=1) / mean_scale > (args.max_valid_scale ** 3)
                            gaussians.clean_up_invalid_gaussians(large_gaussians)
                            
                            print("rendering mesh ...")
                            rendering_mesh(gaussians, training_generator, gaussian_render, args)
                        print("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration, ply_only=True, filename='road_point_cloud.ply')
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration % opt.opacity_reset_interval == 0:
                        torch.cuda.empty_cache()
                        print()
                    if iteration == opt.iterations:
                        progress_bar.close()
                        return

                    # Optimizer step
                    if iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                        if gaussians._opacity.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            gaussians.optimizer.step(relevant)
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    # clamp scaling to prevent gaussians which is too big
                    gaussians._scaling.data.clamp_max_(torch.log(torch.tensor(gaussians.gaussian_road_mean_distance * 4)).cuda())

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
    parser = ArgumentParser(description="Training Road script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--project_dir', required=True, help="project directory")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--load_iteration', type=int, default=-1)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument('--max_valid_scale', type=float, default=3.162)
    parser.add_argument('--generate_novel_views', action='store_true', default=False)
    parser.add_argument("--novel_pos_z", nargs="+", type=int, default=[1])
    parser.add_argument("--novel_rot_z", nargs="+", type=int, default=[0])
    
    args = parser.parse_args(sys.argv[1:])
    args.source_path = os.path.join(args.project_dir, "camera_calibration/rectified")
    args.images = os.path.join(args.source_path, "images")
    args.alpha_masks = os.path.join(args.source_path, "masks")
    args.road_masks = os.path.join(args.source_path, "roadmasks")
    args.model_path = os.path.join(args.project_dir, "output/scaffold")
    args.scaffold_file = os.path.join(args.project_dir, "output/scaffold/point_cloud/iteration_30000")
    args.sh_degree = 1
    args.roadpoints_file = os.path.join(args.source_path, "sparse/roadpoints_dense.ply")
    args.save_iterations.append(args.iterations)
    print("Train Roads Iterations: ", args.iterations, "degree SH: ", args.sh_degree)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(False)
    torch.autograd.set_detect_anomaly(False)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)

    # All done
    print("\nTraining complete.")
