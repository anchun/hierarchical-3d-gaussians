from tqdm import tqdm
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
import argparse
import cv2


def matrix_to_euler_xyz(matrix):
    rotation_matrix = matrix[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('zyx', False)
    translation_vector = matrix[:3, 3]
    return euler_angles, translation_vector

def calculate_w2c(ego_pose, extrinsic):
    OPENCV2DATASET = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    # extrinsic = extrinsic @ OPENCV2DATASET
    c2w = ego_pose @ extrinsic
    rot, pos = c2w[:3,:3], c2w[:3,3]
    rotationInverse = np.linalg.inv(rot)
    pos = -rotationInverse.dot(pos)
    q = Rotation.from_matrix(rotationInverse).as_quat()
    w, x, y, z = q[3], q[0], q[1], q[2]
    return w,x,y,z,pos

if __name__ == '__main__':
    camera_names = ['front_main','left_front','right_front','left_rear','right_rear']
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str, required=True)
    args = parser.parse_args()
    input_notr_dir = args.project_dir
    output_dir = os.path.join(input_notr_dir, 'colmap')
    colmap_modle_path = os.path.join(output_dir, 'sparse/0')
    if not os.path.exists(colmap_modle_path):
        os.makedirs(colmap_modle_path)

    output_img_path = os.path.join(output_dir, 'images')
    for cam_name in camera_names:
        if not os.path.exists(os.path.join(output_img_path, cam_name)):
            os.makedirs(os.path.join(output_img_path, cam_name))
    output_masks_path = os.path.join(output_dir, 'masks')
    for cam_name in camera_names:
        if not os.path.exists(os.path.join(output_masks_path, cam_name)):
            os.makedirs(os.path.join(output_masks_path, cam_name))

    ego_pose_dir = os.path.join(input_notr_dir, 'ego_pose')
    extrinsics_dir = os.path.join(input_notr_dir, 'extrinsics')
    intrinsics_dir = os.path.join(input_notr_dir, 'intrinsics')
    lidar_dir = os.path.join(input_notr_dir, 'lidar')

    # convert extrinsic
    extrinsics = {}
    for filename in os.listdir(extrinsics_dir):
        camera_id = filename.split('.')[0]
        extrinsic = np.loadtxt(os.path.join(extrinsics_dir, filename))
        extrinsics[camera_id] = extrinsic

    widths = {}
    heights = {}
    with open(os.path.join(colmap_modle_path, "images.txt"), 'w') as f:
        i = 1
        for img_filename in tqdm(os.listdir(os.path.join(input_notr_dir, 'images')), desc="Processing images"):
            file_ext = '.' + img_filename.split('.')[1]
            if file_ext not in ['.jpg', '.png', '.jpeg']:
                continue
            frame_cameraId = img_filename.split('.')[0].split('_')
            frame = frame_cameraId[0]
            camera_id = frame_cameraId[1]
            ego_pose = np.loadtxt(os.path.join(ego_pose_dir, frame + '.txt')) 
            if camera_id in extrinsics:
                w,x,y,z,pos = calculate_w2c(ego_pose, extrinsics[camera_id])
                img_name = camera_names[int(camera_id)] + '/' + frame + '_' + camera_id + file_ext
                source_img_path = os.path.join(input_notr_dir, 'images', frame + '_' + camera_id + file_ext)
                dest_img_path = os.path.join(output_img_path, camera_names[int(camera_id)], frame + '_' + camera_id + file_ext)
                if camera_id not in heights or camera_id not in widths:
                    image = cv2.imread(source_img_path)
                    heights[camera_id], widths[camera_id] = image.shape[0], image.shape[1]
                shutil.copyfile(source_img_path, dest_img_path)
                source_mask_path = os.path.join(input_notr_dir, 'dynamic_mask', frame + '_' + camera_id + file_ext)
                dest_mask_path = os.path.join(output_masks_path, camera_names[int(camera_id)], frame + '_' + camera_id + '.png')
                if os.path.exists(source_mask_path):
                    image = cv2.imread(source_mask_path, cv2.IMREAD_GRAYSCALE)
                    inverted_image = 255 - image
                    cv2.imwrite(dest_mask_path, inverted_image)
                f.write('{} {} {} {} {} {} {} {} {} {} \n\n'.format(i,w,x,y,z,pos[0], pos[1], pos[2],camera_id, img_name))
                i = i+1

    # convert intrinsic
    with open(os.path.join(colmap_modle_path, "cameras.txt"), 'w') as f:
        for filename in os.listdir(intrinsics_dir):
            camera_id = filename.split('.')[0]
            intrinsic = np.loadtxt(os.path.join(intrinsics_dir, filename))
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            output = '{} {} {} {} {} {} {} {}'.format(camera_id, 'PINHOLE', widths[camera_id], heights[camera_id], fx, fy, cx, cy)
            f.write(output + '\n')

    with open(os.path.join(colmap_modle_path, "points3D.txt"), 'w') as output:
        pass
