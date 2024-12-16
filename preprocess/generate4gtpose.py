#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os, sys, shutil
import posixpath
import subprocess
import argparse
import time, platform
import numpy as np
from read_write_model import read_images_binary,write_images_binary, Image
from database import COLMAPDatabase
from colmap_helper import update_db_for_colmap_models

def replace_images_by_masks(images_file, out_file):
    """Replace images.jpg to images.png in the colmap images.bin to process masks the same way as images."""
    images_metas = read_images_binary(images_file)
    out_images_metas = {}
    for key in images_metas:
        in_image_meta = images_metas[key]
        out_images_metas[key] = Image(
            id=key,
            qvec=in_image_meta.qvec,
            tvec=in_image_meta.tvec,
            camera_id=in_image_meta.camera_id,
            name=os.path.splitext(in_image_meta.name)[0]+".png",
            xys=in_image_meta.xys,
            point3D_ids=in_image_meta.point3D_ids,
        )
    
    write_images_binary(out_images_metas, out_file)

def setup_dirs(project_dir):
    """Create the directories that will be required."""
    if not os.path.exists(project_dir):
        print("creating project dir.")
        os.makedirs(project_dir)
    
    if not os.path.exists(os.path.join(project_dir, "camera_calibration/aligned")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/aligned/sparse/0"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/rectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/rectified"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified"))
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified", "sparse")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    parser.add_argument('--images_dir', default="", help="Will be set to project_dir/inputs/images if not set")
    parser.add_argument('--masks_dir', default="", help="Will be set to project_dir/inputs/masks if exists and not set")
    parser.add_argument('--depths_dir', default="", help="Will be set to project_dir/inputs/depths if exists and not set")
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "inputs/images")
    if args.masks_dir == "":
        args.masks_dir = os.path.join(args.project_dir, "inputs/masks")
        args.masks_dir = args.masks_dir if os.path.exists(args.masks_dir) else ""
    if args.depths_dir == "":
        args.depths_dir = os.path.join(args.project_dir, "inputs/depths")
        args.depths_dir = args.depths_dir if os.path.exists(args.depths_dir) else ""

    colmap_exe = "colmap"
    if platform.system() == "Windows":
        try:
            subprocess.run([colmap_exe, "-h"], stdout=subprocess.PIPE, check=True)
        except:
            colmap_exe = "colmap.bat"
    glomap_exe = "glomap"
    start_time = time.time()

    print(f"Project will be built here ${args.project_dir} base images are available there ${args.images_dir}.")

    setup_dirs(args.project_dir)

    #init db
    db_filepath = f"{args.project_dir}/camera_calibration/unrectified/database.db"
    model_dir = f"{args.images_dir}/../sparse/0/"
    if not os.path.exists(db_filepath):
        db = COLMAPDatabase.connect(db_filepath)
        db.create_tables()
        # update db from colmap txts which located in the parent folder of images_dir as default
        update_db_for_colmap_models(db, model_dir)
        db.commit()
        db.close()
    
    
    ## Feature extraction, matching then mapper to generate the colmap.
    print("extracting features ...")
    colmap_feature_extractor_args = [
        colmap_exe, "feature_extractor",
        "--database_path", db_filepath,
        "--image_path", f"{args.images_dir}",
        "--ImageReader.single_camera_per_folder", "1",
        "--ImageReader.default_focal_length_factor", "0.5",
        "--ImageReader.camera_model", "OPENCV",
        ]
    
    try:
        subprocess.run(colmap_feature_extractor_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap feature_extractor: {e}")
        sys.exit(1)

    print("making custom matches...")
    make_colmap_custom_matcher_args = [
        "python", f"preprocess/make_colmap_custom_matcher.py",
        "--image_path", f"{args.images_dir}",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt",
        "--n_seq_matches_per_view",f"20"
    ]
    try:
        subprocess.run(make_colmap_custom_matcher_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing make_colmap_custom_matcher: {e}")
        sys.exit(1)

    ## Feature matching
    print("matching features...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", db_filepath,
        "--match_list_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt"
        ]
    try:
        subprocess.run(colmap_matches_importer_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    # triangulate
    os.system(f'colmap point_triangulator \
        --database_path {db_filepath} \
        --image_path {args.images_dir} \
        --input_path {model_dir} \
        --output_path {args.project_dir}/camera_calibration/unrectified/sparse \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.max_extra_param 0 \
        --clear_points 0 \
        --Mapper.ba_global_max_num_iterations 30 \
        --Mapper.filter_max_reproj_error 4 \
        --Mapper.filter_min_tri_angle 0.5 \
        --Mapper.tri_min_angle 0.5 \
        --Mapper.tri_ignore_two_view_tracks 1 \
        --Mapper.tri_complete_max_reproj_error 4 \
        --Mapper.tri_continue_max_angle_error 4')

    ## Undistort images
    print(f"undistorting images from {args.images_dir} to {args.project_dir}/camera_calibration/rectified images...")
    colmap_image_undistorter_args = [
        colmap_exe, "image_undistorter",
        "--image_path", f"{args.images_dir}",
        "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse", 
        "--output_path", f"{args.project_dir}/camera_calibration/rectified/",
        "--output_type", "COLMAP",
        "--max_image_size", "2048",
        ]
    try:
        subprocess.run(colmap_image_undistorter_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    if not args.masks_dir == "":
        # create a copy of colmap as txt and replace jpgs with pngs to undistort masks the same way images were distorted
        if not os.path.exists(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks"):
            os.makedirs(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks")

        shutil.copyfile(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/cameras.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/cameras.bin")
        shutil.copyfile(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/points3D.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/points3D.bin")
        replace_images_by_masks(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/images.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/images.bin")

        print("undistorting masks aswell...")
        colmap_image_undistorter_args = [
            colmap_exe, "image_undistorter",
            "--image_path", f"{args.masks_dir}",
            "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks", 
            "--output_path", f"{args.project_dir}/camera_calibration/tmp/",
            "--output_type", "COLMAP",
            "--max_image_size", "2048",
            ]
        try:
            subprocess.run(colmap_image_undistorter_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing image_undistorter: {e}")
            sys.exit(1)
        
        make_mask_uint8_args = [
            "python", f"preprocess/make_mask_uint8.py",
            "--in_dir", f"{args.project_dir}/camera_calibration/tmp/images",
            "--out_dir", f"{args.project_dir}/camera_calibration/rectified/masks"
        ]
        try:
            subprocess.run(make_mask_uint8_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing make_colmap_custom_matcher: {e}")
            sys.exit(1)

        # remove temporary dir containing undistorted masks
        shutil.rmtree(f"{args.project_dir}/camera_calibration/tmp")

    # copy depths to rectified/depths
    if not args.depths_dir == "":
        depths_target_path = f"{args.project_dir}/camera_calibration/rectified/depths"
        if os.path.exists(depths_target_path):
            shutil.rmtree(depths_target_path)
        shutil.copytree(args.depths_dir, depths_target_path)
    # copy to aligned/sparse/0 without re-orient
    shutil.copyfile(f"{args.project_dir}/camera_calibration/rectified/sparse/images.bin", f"{args.project_dir}/camera_calibration/aligned/sparse/0/images.bin")
    shutil.copyfile(f"{args.project_dir}/camera_calibration/rectified/sparse/cameras.bin", f"{args.project_dir}/camera_calibration/aligned/sparse/0/cameras.bin")
    shutil.copyfile(f"{args.project_dir}/camera_calibration/rectified/sparse/points3D.bin", f"{args.project_dir}/camera_calibration/aligned/sparse/0/points3D.bin")

    end_time = time.time()
    print(f"Preprocessing done in {(end_time - start_time)/60.0} minutes.")
