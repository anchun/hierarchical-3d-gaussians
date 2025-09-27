import json
import logging
import os
import shutil
import sys
import argparse
import cv2
import numpy as np
import random
import open3d as o3d

from pathlib import Path
from scipy.spatial.transform import Rotation

camera_names = ['left_rear','left_front','front_main','right_front','right_rear']
camera_indexs={'front_main':0,'left_front':1,'right_front':2,'left_rear':3,'right_rear':4}
camera_name_2_id = {c:i for i,c in enumerate(camera_names)}

def read_camera_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        cameras = {}

        for i, line in enumerate(lines):
            if '#' in line:
                continue;
            tmp = line.split(' ')
            camera_id = int(tmp[0])
            if i < 200:
                fx,fy,cx,cy = float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                D = np.array([0, 0, 0, 0])

                cameras[camera_id] = {'K':K,'D':D}

        return cameras
    
def undistorted_image_and_intrinsics(spath):
    cameras = read_camera_txt(os.path.join(spath, 'colmap/sparse/0/cameras.txt'))
    distorted_intrinsics = {}
    distorted_w = {}
    distorted_h = {}
    for camera_name in camera_names:
        cindex = camera_indexs[camera_name]

        K = cameras.get(cindex)['K']
        camera_image_path = os.path.join(spath, 'colmap/images', camera_name)
        camera_mask_path = os.path.join(spath, 'colmap/masks', camera_name)
        camera_output_image_path = os.path.join(spath, 'inputs/images', camera_name)
        os.makedirs(camera_output_image_path, exist_ok=True)
        camera_output_masks_path = os.path.join(spath, 'inputs/masks', camera_name)
        os.makedirs(camera_output_masks_path, exist_ok=True)
        image_names = os.listdir(camera_image_path)
        for filename in image_names:
            image = cv2.imread(os.path.join(camera_image_path, filename))
            # undistort
            print("undistorting only with cropping: ", os.path.join(camera_image_path, filename))
            h, w = image.shape[:2]
            # crop
            x, y = 0, 0
            cx, cy = K[0, 2], K[1, 2]
            new_w, new_h = round(max(cx - x, x + w - cx)).__int__() * 2, round(max(cy - y, y + h - cy)).__int__() * 2
            new_x, new_y = round(new_w / 2 - cx).__int__(), round(new_h / 2 - cy).__int__()
            new_x, new_y = max(new_x, 0), max(new_y, 0)
            newK = K.copy()
            newK[0, 2], newK[1, 2] = new_w / 2, new_h / 2
            image_out = np.zeros((new_h, new_w, 3), dtype=image.dtype)
            image_out[new_y:new_y + h, new_x:new_x + w] = image
            # output image
            distorted_intrinsics[camera_name] = newK
            distorted_w[camera_name] = new_w
            distorted_h[camera_name] = new_h
            cv2.imwrite(os.path.join(camera_output_image_path, filename), image_out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # output mask
            mask_filename = Path(filename).with_suffix(".png")
            if os.path.exists(os.path.join(camera_mask_path, mask_filename)):
                mask = cv2.imread(os.path.join(camera_mask_path, mask_filename))
                mask_out = np.zeros((new_h, new_w, 3), dtype=mask.dtype)
                mask_out[new_y:new_y + h, new_x:new_x + w] = mask
                cv2.imwrite(os.path.join(camera_output_masks_path, mask_filename), mask_out)

    # sparse
    camera_output_sparse_path = os.path.join(spath, 'inputs/sparse/0')
    os.makedirs(camera_output_sparse_path, exist_ok=True)
    shutil.copyfile(os.path.join(spath, 'colmap/sparse/0/images.txt'), os.path.join(camera_output_sparse_path, 'images.txt'))
    Path(os.path.join(camera_output_sparse_path, 'points3D.txt')).touch()
    with open(os.path.join(camera_output_sparse_path, 'cameras.txt'), 'w') as output:
        for i,camera_name in enumerate(camera_names):
            index = camera_indexs[camera_name]
            intrinsic = distorted_intrinsics[camera_name]
            output.write(str(index) + ' PINHOLE ' + str(distorted_w[camera_name]) + ' ' + str(distorted_h[camera_name])
                         + ' ' + str(intrinsic[0][0]) + ' ' + str(intrinsic[1][1]) + ' ' + str(intrinsic[0][2]) + ' ' + str(intrinsic[1][2]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    args = parser.parse_args()
    undistorted_image_and_intrinsics(args.project_dir)



