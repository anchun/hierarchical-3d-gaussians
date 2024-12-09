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

import numpy as np

from database import COLMAPDatabase
from read_write_model import read_images_binary,write_images_binary, Image
import time, platform
camModelDict = {'SIMPLE_PINHOLE': 0,
                'PINHOLE': 1,
                'SIMPLE_RADIAL': 2,
                'RADIAL': 3,
                'OPENCV': 4,
                'OPENCV_FISHEYE':5,
                'FULL_OPENCV':6,
                'FOV': 7,
                'SIMPLE_RADIAL_FISHEYE' :8,
                'RADIAL_FISHEYE': 9,
                'THIN_PRISM_FISHEYE': 10}

def get_init_cameraparams(camera_params, modelId=0):
    fx = camera_params["fx"]
    fy = camera_params["fy"]
    f = (fx + fy) / 2.0
    cx = camera_params["cx"]
    cy = camera_params["cy"]

    if modelId == 0:
        return np.array([f, cx, cy])
    elif modelId == 1:
        return np.array([fx, fy, cx, cy])
    elif modelId == 2 or modelId == 8:
        return np.array([f, cx, cy, 0.0])
    elif modelId == 3 or modelId == 7:
        return np.array([f, cx, cy, 0.0, 0.0])
    elif modelId == 4 or modelId == 5:
        return np.array([(fx, fy, cx, cy, 0.0)])
    elif modelId == 9:
        return np.array([fx, fy, cx, cy, 0.0])
    return np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def update_db_for_colmap_models(db, model_path):
    # read camera info
    camera_ids = []
    camera_images = {}
    cameras_file = posixpath.join(model_path, "cameras.txt")
    with open(cameras_file, "r") as fid:
        for line in fid:
            if line.startswith('#'):
                continue
            values = line.strip().split(' ')
            camera_model = values[1]
            if camera_model == 'PINHOLE':
                camera_id, w, h, fx, fy, cx, cy = int(values[0]), int(values[2]), int(values[3]), float(
                    values[4]), float(values[5]), float(values[6]), float(values[7])
            elif camera_model == 'SIMPLE_PINHOLE':
                camera_id, w, h, fx, fy, cx, cy = int(values[0]), int(values[2]), int(values[3]), float(
                    values[4]), float(values[4]), float(values[5]), float(values[6])
            else:
                print("invalid camera!")
                sys.exit(-1)
            # save cacmera db
            camera_params = {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
            params = get_init_cameraparams(camera_params, camModelDict[camera_model])
            print("Camera type: ", camera_model, "Width: ", w, "Height: ", h)
            print("Camera parameters: ", params)
            cx_ratio = abs(cx * 2 / w - 1)
            cy_ratio = abs(cy * 2 / h - 1)
            if cx_ratio > 0.1 or cy_ratio > 0.1:
                print("cx or cy value invalid!")
                sys.exit(-1)
            camera_ids.append(camera_id)
            camera_images[camera_id] = []
            db.add_camera(camModelDict[camera_model], w, h, params, prior_focal_length=True, camera_id=camera_id)

    # read image info.
    images_file = posixpath.join(model_path, "images.txt")

    count = 0
    with open(images_file, "r") as fid:
        for line in fid:
            if line.startswith('#'):
                continue
            values = line.strip().split(' ')
            if len(values) < 10:
                continue
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name = int(values[0]), float(values[1]), float(
                values[2]), float(values[3]), float(values[4]), float(values[5]), float(values[6]), float(
                values[7]), int(values[8]), values[9]
            # print(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name)
            camera_images[camera_id].append(
                {'image_name': image_name, 'prior_q': [qw, qx, qy, qz], 'prior_t': [tx, ty, tz]})
            count += 1

    # add db and update image_id with new sequence
    count = 0
    for camera_id in camera_ids:
        for camera_image in camera_images[camera_id]:
            camera_image['image_id'] = count
            db.add_image(camera_image['image_name'], camera_id, camera_image['prior_q'], camera_image['prior_t'],
                         image_id=camera_image['image_id'])
            count += 1

    # refine image txt with new id
    if len(camera_images) > 0:
        HEADER = '# Images list with two lines of data per image:n' + \
                 "#   Images_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
                 "#   POINTS2D[] AS (X, Y, POINT3D_ID)\n" + \
                 "# Number of images: {}, mean obersevation per image: {}\n".format(count, 0)
        with open(images_file, "w") as fid:
            fid.write(HEADER)
            for idx in range(len(camera_images[0])):
                for camera_id in camera_ids:
                    camera_image = camera_images[camera_id][idx]
                    image_header = [camera_image['image_id'], camera_image['prior_q'][0], camera_image['prior_q'][1],
                                    camera_image['prior_q'][2], camera_image['prior_q'][3],
                                    camera_image['prior_t'][0], camera_image['prior_t'][1], camera_image['prior_t'][2],
                                    camera_id, camera_image['image_name']]
                    first_line = " ".join(map(str, image_header))
                    fid.write(first_line + "\n")
                    fid.write("\n")
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
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "inputs/images")
    if args.masks_dir == "":
        args.masks_dir = os.path.join(args.project_dir, "inputs/masks")
        args.masks_dir = args.masks_dir if os.path.exists(args.masks_dir) else ""

    colmap_exe = "colmap"
    glomap_exe = "glomap"
    start_time = time.time()

    print(f"Project will be built here ${args.project_dir} base images are available there ${args.images_dir}.")

    setup_dirs(args.project_dir)

    #init db
    db_filepath = f"{args.project_dir}/camera_calibration/unrectified/database.db"
    if not os.path.exists(db_filepath):
        db = COLMAPDatabase.connect(db_filepath)
        db.create_tables()
        # update db from colmap txts which located in the parent folder of images_dir as default
        update_db_for_colmap_models(db, f"{args.images_dir}/../")
        db.commit()
        db.close()
    
    
    # Feature extraction, matching then mapper to generate the colmap.
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
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--match_list_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt"
        ]
    try:
        subprocess.run(colmap_matches_importer_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    ## Generate sfm pointcloud  with glomap
    print("generating sfm point cloud with glomap...")
    glomap_mapper_args = [
        glomap_exe, "mapper",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--image_path", f"{args.images_dir}",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/",
        "--TrackEstablishment.max_num_tracks", "50000",
        "--Triangulation.complete_max_reproj_error","5",
        "--Triangulation.merge_max_reproj_error","5",
        "--Triangulation.min_num_matches","100"
        ]
    try:
        subprocess.run(glomap_mapper_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing glomap mapper: {e}")
        sys.exit(1)

    ## Simplify images so that everything takes less time (reading colmap usually takes forever)
    simplify_images_args = [
        "python", f"preprocess/simplify_images.py",
        "--base_dir", f"{args.project_dir}/camera_calibration/unrectified/sparse/0"
    ]
    try:
        subprocess.run(simplify_images_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing simplify_images: {e}")
        sys.exit(1)

    ## Undistort images
    print(f"undistorting images from {args.images_dir} to {args.project_dir}/camera_calibration/rectified images...")
    colmap_image_undistorter_args = [
        colmap_exe, "image_undistorter",
        "--image_path", f"{args.images_dir}",
        "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0", 
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

    # re-orient + scale colmap
    print(f"re-orient and scaling scene to {args.project_dir}/camera_calibration/aligned/sparse/0")
    reorient_args = [
            "python", f"preprocess/auto_reorient.py",
            "--input_path", f"{args.project_dir}/camera_calibration/rectified/sparse",
            "--output_path", f"{args.project_dir}/camera_calibration/aligned/sparse/0"
        ]
    try:
        subprocess.run(reorient_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing auto_orient: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Preprocessing done in {(end_time - start_time)/60.0} minutes.")
