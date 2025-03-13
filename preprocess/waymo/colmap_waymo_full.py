import sqlite3
import os
import numpy as np
import glob
import os
import cv2
import shutil
import sys
sys.path.append(os.getcwd())
import json

from scipy.spatial.transform import Rotation as R
from waymo_utils import load_camera_info
from scene.colmap_loader import read_extrinsics_binary, qvec2rotmat

image_filename_to_cam = lambda x: int(x.split('/')[0].split('_')[1]) # cam_{cam_id}/{frame}.jpg

def convert_filename(filename):
    # {frame}_{cam_id}.jpg -> cam_{cam_id}/{frame}.jpg
    frame, cam_id = filename.split('.')[0].split('_')
    new_filename = f'cam_{cam_id}/{frame}.jpg'
    return new_filename

def run_colmap_waymo(input_dir, output_dir, result):
    colmap_dir = output_dir
    os.makedirs(colmap_dir, exist_ok=True)
    print('running colmap, colmap dir: ', colmap_dir)

    unique_cams = sorted(list(set(result['cams'])))
    print('cameras: ', unique_cams)
    for unqiue_cam in unique_cams:
        train_images_dir = os.path.join(colmap_dir, 'images', f'cam_{unqiue_cam}')
        test_images_dir = os.path.join(colmap_dir, 'test_imgs', f'cam_{unqiue_cam}')
        mask_images_dir = os.path.join(colmap_dir, 'masks', f'cam_{unqiue_cam}')
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)
        os.makedirs(mask_images_dir, exist_ok=True)
    
    train_images_dir = os.path.join(colmap_dir, 'images')
    test_images_dir = os.path.join(colmap_dir, 'test_imgs')
    mask_images_dir = os.path.join(colmap_dir, 'masks')
    
    image_filenames = result['image_filenames']
    c2ws = result['c2ws']
    ixts = result['ixts']
    frames_idx = result['frames_idx']
    cams = result['cams']

    # load extrinsic and image filenames
    c2w_dict = dict()
    train_image_filenames = []
    mask_image_filenames = []
    result['new_image_filenames'] = []
    for i, image_filename in enumerate(image_filenames):
        frame_idx = frames_idx[i]
        basename = os.path.basename(image_filename)
        new_image_filename = convert_filename(basename)
        c2w_dict[new_image_filename] = c2ws[i]
        mask_image_filenames.append(os.path.join(input_dir, 'dynamic_mask', basename))
        train_image_filenames.append(image_filename)
        result['new_image_filenames'].append(new_image_filename)

    # copy images
    for i, image_filename in enumerate(train_image_filenames):
        basename = os.path.basename(image_filename)
        new_image_filename = os.path.join(train_images_dir, convert_filename(basename))
        if not os.path.exists(new_image_filename):
            shutil.copyfile(image_filename, new_image_filename)

    # copy mask
    for i, image_filename in enumerate(mask_image_filenames):
        basename = os.path.basename(image_filename)
        mask_images_dir = os.path.join(colmap_dir, 'masks')
        new_image_filename = os.path.join(mask_images_dir, convert_filename(basename))
        new_mask_filename = f'{new_image_filename}'
        if not os.path.exists(new_mask_filename):
            shutil.copyfile(image_filename, new_mask_filename)
            mask = cv2.imread(new_mask_filename)
            flip_mask = (255 - mask).astype(np.uint8)
            cv2.imwrite(new_mask_filename, flip_mask)
    
    # https://colmap.github.io/faq.html#mask-image-regions
    os.system(f'colmap feature_extractor \
            --ImageReader.mask_path {mask_images_dir} \
            --ImageReader.camera_model SIMPLE_PINHOLE  \
            --ImageReader.single_camera_per_folder 1 \
            --database_path {colmap_dir}/database.db \
            --image_path {train_images_dir}')

    # load intrinsic
    camera_infos = dict()
    for unique_cam in unique_cams:
        for i, cam in enumerate(cams):
            if cam == unique_cam: 
                break
        sample_img = cv2.imread(image_filenames[i])
        img_h, img_w = sample_img.shape[:2]
        camera_infos[unique_cam] = {
            'ixt': ixts[i],
            'img_h': img_h,
            'img_w': img_w,
        }

    # load id_names from database
    db = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()
        
    out_fn = f'{colmap_dir}/id_names.txt'
    with open(out_fn, 'w') as f:
        for i in result:
            f.write(str(i[0]) + ' ' + i[1] + '\n')
    f.close()

    path_idname = f'{colmap_dir}/id_names.txt'
    
    f_id_name = open(path_idname, 'r')
    f_id_name_lines= f_id_name.readlines()

    model_dir = f'{colmap_dir}/created/sparse'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # create images.txt
    f_w = open(f'{model_dir}/images.txt', 'w')
    id_names = []
    for l in f_id_name_lines:
        l = l.strip().split(' ')
        id_ = int(l[0])
        name = l[1]
        id_names.append([id_, name])

    for i in range(len(id_names)):
        id_ = id_names[i][0]
        name = id_names[i][1]
        transform = c2w_dict[name]
        transform = np.linalg.inv(transform)

        r = R.from_matrix(transform[:3,:3])
        rquat = r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
        rquat[0], rquat[1], rquat[2], rquat[3] = rquat[3], rquat[0], rquat[1], rquat[2]
        out = np.concatenate((rquat, transform[:3, 3]), axis=0)

        id_ = id_names[i][0]
        name = id_names[i][1]
        cam = image_filename_to_cam(name)

        f_w.write(f'{id_} ')
        f_w.write(' '.join([str(a) for a in out.tolist()] ) )
        f_w.write(f' {cam} {name}')
        f_w.write('\n\n')
    
    f_w.close()

    # create cameras.txt
    cameras_fn = os.path.join(model_dir, 'cameras.txt')
    with open(cameras_fn, 'w') as f:
        for unique_cam in unique_cams:
            camera_info = camera_infos[unique_cam]
            ixt = camera_info['ixt']
            img_w = camera_info['img_w']
            img_h = camera_info['img_h']
            fx = ixt[0, 0]
            fy = ixt[1, 1]
            cx = ixt[0, 2]
            cy = ixt[1, 2]
            f.write(f'{unique_cam} SIMPLE_PINHOLE {img_w} {img_h} {fx} {cx} {cy}')
            f.write('\n')

    # update database
    db = f'{colmap_dir}/database.db'
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('SELECT * FROM images')
    result = c.fetchall()
    cam_to_id = dict()
    for i in result:
        name = i[1]
        cam = image_filename_to_cam(name)
        cam_id = i[2]
        cam_to_id[cam] = cam_id
    
    for unique_cam in unique_cams:
        cam_id = cam_to_id[unique_cam]
        ixt = camera_infos[unique_cam]['ixt']
        fx, fy, cx, cy = ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]
        params = np.array([fx, cx, cy]).astype(np.float64)
        c.execute("UPDATE cameras SET params = ? WHERE camera_id = ?",
                  (params.tostring(), cam_id))
    conn.commit()
    conn.close()

    # create points3D.txt
    points3D_fn = os.path.join(model_dir, 'points3D.txt')
    os.system(f'touch {points3D_fn}')
    
    # create rid ba config
    cam_rigid = dict()

    ref_camera_id = unique_cams[0]
    cam_rigid["ref_camera_id"] = ref_camera_id
    rigid_cam_list = []

    _, extrinsics, _, _ = load_camera_info(input_dir)
    for cam_id in unique_cams:
        rigid_cam = dict()
        rigid_cam["camera_id"] = cam_id

        ref_extrinsic = extrinsics[ref_camera_id]
        cur_extrinsic = extrinsics[cam_id]
        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic
        # print('relative extrinisc')
        # print(cam_id, rel_extrinsic)
        r = R.from_matrix(rel_extrinsic[:3, :3])
        qvec = r.as_quat()
        rigid_cam["image_prefix"] = 'cam_{}'.format(cam_id)        
        rigid_cam['cam_from_rig_rotation'] = [qvec[3], qvec[0], qvec[1], qvec[2]]
        rigid_cam['cam_from_rig_translation'] = [rel_extrinsic[0, 3], rel_extrinsic[1, 3], rel_extrinsic[2, 3]]
        
        rigid_cam_list.append(rigid_cam)

    cam_rigid["cameras"] = rigid_cam_list

    rigid_config_path = os.path.join(colmap_dir, "cam_rigid_config.json")
    with open(rigid_config_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)   

    os.system(f'colmap exhaustive_matcher \
            --database_path {colmap_dir}/database.db')

    triangulated_dir = os.path.join(colmap_dir, 'sparse')
    os.makedirs(triangulated_dir, exist_ok=True)
    os.system(f'colmap point_triangulator \
        --database_path {colmap_dir}/database.db \
        --image_path {train_images_dir} \
        --input_path {model_dir} \
        --output_path {triangulated_dir} \
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
    
    if False: # cfg.data.use_colmap_pose: TODO
        # May lead to unstable results when refining relative poses
        os.system(f'colmap rig_bundle_adjuster \
                --input_path {triangulated_dir} \
                --output_path {triangulated_dir} \
                --rig_config_path {rigid_config_path} \
                --estimate_rig_relative_poses 0 \
                --RigBundleAdjustment.refine_relative_poses 1 \
                --BundleAdjustment.max_num_iterations 50 \
                --BundleAdjustment.refine_focal_length 0 \
                --BundleAdjustment.refine_principal_point 0 \
                --BundleAdjustment.refine_extra_params 0')

    #os.system(f'rm -rf {train_images_dir}')
    #os.system(f'rm -rf {test_images_dir}')  
    #os.system(f'rm -rf {mask_images_dir}')
    
if __name__ == '__main__':
    run_colmap_waymo(result=None)
