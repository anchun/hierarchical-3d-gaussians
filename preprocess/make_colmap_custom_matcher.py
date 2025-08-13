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

import os
import numpy as np
from joblib import delayed, Parallel
import argparse
import math
from exif import Image
from numpy import sort
from sklearn.neighbors import NearestNeighbors
from database import COLMAPDatabase, blob_to_array

def get_all_images(db: COLMAPDatabase):
    images_list = {}
    camera_images_list = {}
    entries = db.execute("SELECT images.image_id, name, camera_id, position FROM images, pose_priors where images.image_id = pose_priors.image_id")
    for image_id, name, camera_id, position in entries:
        images_list[name] = {"image_id": image_id, "camera_id": camera_id, "position": blob_to_array(position, np.float64)}
        camera_images_list.setdefault(camera_id, []).append(name)
    return images_list, camera_images_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--database_path', required=True)
    parser.add_argument('--n_seq_matches_per_view', default=20, type=int)
    parser.add_argument('--n_quad_matches_per_view', default=0, type=int)
    parser.add_argument('--with_camera_loop', action="store_true", default=False)
    parser.add_argument('--n_pose_neighbours', default=0, type=int)
    args = parser.parse_args()

    db = COLMAPDatabase.connect(str(args.database_path))
    images_list, image_files_organised = get_all_images(db)
    db.close()

    matches_str = []
    def add_match(current_image_file, matched_cam, matched_frame_id):
        if matched_frame_id < len(matched_cam):
            matched_image_file = matched_cam[matched_frame_id]
            if current_image_file != matched_image_file:
                matches_str.append(f"{current_image_file} {matched_image_file}\n")


    for cam_id, current_cam in image_files_organised.items():
        for matched_cam_id, matched_cam in image_files_organised.items():
            id_diff = math.fabs(cam_id - matched_cam_id)
            if id_diff == 0 or id_diff == 1 or (args.with_camera_loop and id_diff == len(image_files_organised) - 1):
                for current_image_id, current_image_file in enumerate(current_cam):
                    for frame_step in range(args.n_seq_matches_per_view):
                        matched_frame_id = current_image_id + frame_step
                        add_match(current_image_file, matched_cam, matched_frame_id)

                    for match_id in range(args.n_quad_matches_per_view):
                        frame_step = args.n_seq_matches_per_view + int(2**match_id) - 1
                        matched_frame_id = current_image_id + frame_step
                        add_match(current_image_file, matched_cam, matched_frame_id)


    ## Add Pose matches
    if args.n_pose_neighbours > 0:
        all_img_names = list(images_list.keys())
        all_cam_centers = [images_list[img_name]['position'] for img_name in all_img_names]

        cam_centers = np.array(all_cam_centers, dtype=np.float64)
        cam_nbrs = NearestNeighbors(n_neighbors=args.n_pose_neighbours).fit(cam_centers) if cam_centers.size else []

        for img_name, cam_center in zip(all_img_names, cam_centers):
            _, indices = cam_nbrs.kneighbors(cam_center[None])
            for idx in indices[0, 1:]:
                matches_str.append(f"{img_name} {all_img_names[idx]}\n")

    ## Remove duplicate matches
    print(f"Total matches before removing duplicates: {len(matches_str)}")
    matches_dict = {}
    for match in matches_str:
        match_first = match.split(' ')[0]
        match_second = match.split(' ')[1][:-1]
        reverse_match = f"{match_second} {match_first}\n"
        if match_first != match_second and match not in matches_dict and reverse_match not in matches_dict:
            matches_dict[match] = 1
    
    out_matches = list(matches_dict.keys())
    print(f"Total matches after removing duplicates: {len(out_matches)}")
    with open(args.output_path, "w") as f:
        f.write(''.join(out_matches))

    print(0)