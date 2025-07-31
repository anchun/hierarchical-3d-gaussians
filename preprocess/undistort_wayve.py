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

camera_names = ['front-forward','left-forward','right-forward','left-backward','right-backward']
camera_indexs={'front-forward':1,'left-forward':8,'right-forward':10,'left-backward':7,'right-backward':9}
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
                k1,k2,k3,k4 = float(tmp[8]), float(tmp[9]), float(tmp[10]), float(tmp[11])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                D = np.array([k1, k2, k3, k4])

                cameras[camera_id] = {'K':K,'D':D}

        return cameras
    
def undistorted_image_and_intrinsics(spath):
    cameras = read_camera_txt(os.path.join(spath, 'colmap_sparse/cameras.txt'))
    distorted_intrinsics = {}
    distorted_w = {}
    distorted_h = {}
    for camera_name in camera_names:
        cindex = camera_indexs[camera_name]

        K = cameras.get(cindex)['K']
        D = cameras.get(cindex)['D']
        camera_image_path = os.path.join(spath, 'images', camera_name)
        camera_mask_path = os.path.join(spath, 'masks', camera_name)
        camera_output_image_path = os.path.join(spath, 'inputs/images', camera_name)
        os.makedirs(camera_output_image_path, exist_ok=True)
        camera_output_masks_path = os.path.join(spath, 'inputs/masks', camera_name)
        os.makedirs(camera_output_masks_path, exist_ok=True)
        image_names = os.listdir(camera_image_path)
        for filename in image_names:
            image = cv2.imread(os.path.join(camera_image_path, filename))
            # undistort
            print("undistorting: ", os.path.join(camera_image_path, filename))
            h, w = image.shape[:2]
            newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1)
            undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
            # crop
            x, y = 0, 0
            cx, cy = newK[0, 2], newK[1, 2]
            w, h = round(min(cx - x, x + w - cx) * 2).__int__(), round(min(cy - y, y + h - cy) * 2).__int__()
            x, y = round(cx - w / 2).__int__(), round(cy - h / 2).__int__()
            newK[0, 2], newK[1, 2] = w / 2, h / 2
            undistorted_image = undistorted_image[y:y + h, x:x + w]
            # output image
            distorted_intrinsics[camera_name] = newK
            distorted_w[camera_name] = w
            distorted_h[camera_name] = h
            cv2.imwrite(os.path.join(camera_output_image_path, filename), undistorted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # output mask
            mask_filename = Path(filename).with_suffix(".png")
            if os.path.exists(os.path.join(camera_mask_path, mask_filename)):
                mask = cv2.imread(os.path.join(camera_mask_path, mask_filename))
                undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_NEAREST)
                undistorted_mask = undistorted_mask[y:y + h, x:x + w]
                cv2.imwrite(os.path.join(camera_output_masks_path, mask_filename), undistorted_mask)

    # sparse
    camera_output_sparse_path = os.path.join(spath, 'inputs/sparse/0')
    os.makedirs(camera_output_sparse_path, exist_ok=True)
    shutil.copyfile(os.path.join(spath, 'colmap_sparse/images.txt'), os.path.join(camera_output_sparse_path, 'images.txt'))
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



