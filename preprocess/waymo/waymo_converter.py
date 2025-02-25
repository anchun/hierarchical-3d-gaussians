import sys
import os
sys.path.append(os.getcwd())
import torch
import numpy as np
import cv2
import math
# import imageio
import argparse
import json
from tqdm import tqdm
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
from waymo_utils import generate_dataparser_outputs
from utils.box_utils import bbox_to_corner3d, get_bound_2d_mask
import joblib


# castrack_path = '/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json'
# with open(castrack_path, 'r') as f:
#     castrack_infos = json.load(f)

camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT',
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT',
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}

image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]

laser_names_dict = {
    dataset_pb2.LaserName.TOP: 'TOP',
    dataset_pb2.LaserName.FRONT: 'FRONT',
    dataset_pb2.LaserName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.LaserName.SIDE_RIGHT: 'SIDE_RIGHT',
    dataset_pb2.LaserName.REAR: 'REAR',
}

opencv2camera = np.array([[0., 0., 1., 0.],
                        [-1., 0., 0., 0.],
                        [0., -1., 0., 0.],
                        [0., 0., 0., 1.]])


def project_numpy(xyz, K, RT, H, W):
    '''
    input:
    xyz: [N, 3], pointcloud
    K: [3, 3], intrinsic
    RT: [4, 4], w2c

    output:
    mask: [N], pointcloud in camera frustum
    xy: [N, 2], coord in image plane
    '''

    xyz_cam = np.dot(xyz, RT[:3, :3].T) + RT[:3, 3:].T
    valid_depth = xyz_cam[:, 2] > 0
    xyz_pixel = np.dot(xyz_cam, K.T)
    xyz_pixel = xyz_pixel[:, :2] / xyz_pixel[:, 2:]
    valid_x = np.logical_and(xyz_pixel[:, 0] >= 0, xyz_pixel[:, 0] < W)
    valid_y = np.logical_and(xyz_pixel[:, 1] >= 0, xyz_pixel[:, 1] < H)
    valid_pixel = np.logical_and(valid_x, valid_y)
    mask = np.logical_and(valid_depth, valid_pixel)

    return xyz_pixel, mask


def draw_3d_box_on_img(vertices, img, color=(255, 128, 128), thickness=1):
    # Draw the edges of the 3D bounding box
    for k in [0, 1]:
        for l in [0, 1]:
            for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)

    # Draw a cross on the front face to identify front & back.
    for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
        cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness)


def get_extrinsic(camera_calibration):
    camera_extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4) # camera to vehicle
    extrinsic = np.matmul(camera_extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
    return extrinsic
    
def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic

def project_label_to_image(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    points_uv, valid = project_numpy(
        xyz=points_vehicle[..., :3], 
        K=intrinsic, 
        RT=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    return points_uv, valid

def project_label_to_mask(dim, obj_pose, calibration):
    bbox_l, bbox_w, bbox_h = dim
    bbox = np.array([[-bbox_l, -bbox_w, -bbox_h], [bbox_l, bbox_w, bbox_h]]) * 0.5
    points = bbox_to_corner3d(bbox)
    points = np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)
    points_vehicle = points @ obj_pose.T # 3D bounding box in vehicle frame
    extrinsic = get_extrinsic(calibration)
    intrinsic = get_intrinsic(calibration)
    width, height = calibration.width, calibration.height
    mask = get_bound_2d_mask(
        corners_3d=points_vehicle[..., :3],
        K=intrinsic,
        pose=np.linalg.inv(extrinsic), 
        H=height, W=width
    )
    
    return mask
    
    
def parse_seq_rawdata(process_list, root_dir, seq_name, seq_save_dir, track_file, start_idx=None, end_idx=None):
    print(f'Processing sequence {seq_name}...')
    print(f'Saving to {seq_save_dir}')
    all_obj_transformers = {}
    all_obj_rotation_matrixes = {}
    cam_extrinsics = None
    try:
        with open(track_file, 'r') as f:
            castrack_infos = json.load(f)
    except:
        castrack_infos = dict()

    os.makedirs(seq_save_dir, exist_ok=True)
    
    seq_path = os.path.join(root_dir, seq_name+'.tfrecord')
    
    # set start and end timestep
    datafile = WaymoDataFileReader(seq_path)
    num_frames = len(datafile.get_record_table())
    start_idx = start_idx or 0
    end_idx = end_idx or num_frames - 1
    
    if 'pose' in process_list:
        ego_pose_save_dir = os.path.join(seq_save_dir, 'ego_pose')
        os.makedirs(ego_pose_save_dir, exist_ok=True)
        print("Processing ego pose...")
        timestamp = dict()
        timestamp['FRAME'] = dict()
        for camera_name in camera_names_dict.values():
            timestamp[camera_name] = dict()
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            pose = np.array(frame.pose.transform).reshape(4, 4)
            np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}.txt"), pose)
            timestamp['FRAME'][str(frame_id).zfill(6)] = frame.timestamp_micros / 1e6
            
            camera_calibrations = frame.context.camera_calibrations
            for i, camera in enumerate(camera_calibrations):
                camera_name = camera.name
                camera_name_str = camera_names_dict[camera_name]
                camera = utils.get(frame.images, camera_name)
                camera_timestamp = camera.pose_timestamp
                timestamp[camera_name_str][str(frame_id).zfill(6)] = camera_timestamp
                
                camera_pose = np.array(camera.pose.transform).reshape(4, 4)
                np.savetxt(os.path.join(ego_pose_save_dir, f"{str(frame_id).zfill(6)}_{camera_name-1}.txt"), camera_pose)

        timestamp_save_path = os.path.join(seq_save_dir, "timestamps.json")
        with open(timestamp_save_path, 'w') as f:
            json.dump(timestamp, f, indent=1)

    
    if 'calib' in process_list:
        intrinsic_save_dir = os.path.join(seq_save_dir, 'intrinsics')
        extrinsic_save_dir = os.path.join(seq_save_dir, 'extrinsics')
        os.makedirs(intrinsic_save_dir, exist_ok=True)
        os.makedirs(extrinsic_save_dir, exist_ok=True)
        print("Processing camera calibration...")
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            camera_calibrations = frame.context.camera_calibrations
        
        extrinsics = []
        intrinsics = []
        camera_names = []
        for camera in camera_calibrations:
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            extrinsic = np.matmul(extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
            intrinsic = list(camera.intrinsic)
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            camera_names.append(camera.name)
        
        for i in range(5):
            np.savetxt(os.path.join(extrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), extrinsics[i])
            np.savetxt(os.path.join(intrinsic_save_dir, f"{str(camera_names[i] - 1)}.txt"), intrinsics[i])
        cam_extrinsics = extrinsics
    
    if 'image' in process_list:
        image_save_dir = os.path.join(seq_save_dir, 'images')
        os.makedirs(image_save_dir, exist_ok=True)        
        print("Processing image data...")
        
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            for camera_name, camera_name_str in camera_names_dict.items():                    
                camera = utils.get(frame.images, camera_name)
                img = utils.decode_image(camera)
                img_path = os.path.join(image_save_dir, f'{frame_id:06d}_{str(camera.name - 1)}.png')
                cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print("Processing image data done...")

    if 'lidar' in process_list:
        pts_3d_all = dict()
        pts_2d_all = dict()
        print("Processing LiDAR data...")
                
        datafile = WaymoDataFileReader(seq_path)
        for frame_id, frame in tqdm(enumerate(datafile)):
            pts_3d = [] # LiDAR point cloud in world frame
            pts_2d = [] # LiDAR point cloud projection in camera [camera_name, w, h] 
            
            for laser_name, laser_name_str in laser_names_dict.items():
                laser = utils.get(frame.lasers, laser_name)
                laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)
                ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

                # LiDAR spherical coordinate -> polar -> cartesian
                pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
                pts_3d.append(pcl[:, :3]) # save LiDAR pointcloud in vehicle frame 
                
                # Transform LIDAR point cloud from vehicle frame to world frame
                # vehicle_pose = np.array(frame.pose.transform).reshape(4, 4)
                # pcl = vehicle_pose.dot(np.concatenate([pcl, np.ones((pcl.shape[0], 1))], axis=1).T).T
                                    
                mask = ri[:, :, 0] > 0
                camera_projection = camera_projection[mask]

                # Can be projected to multi-cameras, order: [FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT].
                # Only save the first projection camera,

                # camera_projection
                # Inner dimensions are:
                # channel 0: CameraName.Name of 1st projection. Set to UNKNOWN if no projection.
                # channel 1: x (axis along image width)
                # channel 2: y (axis along image height)
                # channel 3: CameraName.Name of 2nd projection. Set to UNKNOWN if no projection.
                # channel 4: x (axis along image width)
                # channel 5: y (axis along image height)
                # Note: pixel 0 corresponds to the left edge of the first pixel in the image.
                
                camera_projection[:, 0] -= 1
                camera_projection[:, 3] -= 1
                camera_projection = camera_projection.astype(np.int16)
                
                pts_2d.append(camera_projection)

            pts_3d = np.concatenate(pts_3d, axis=0)
            pts_3d_all[frame_id] = pts_3d
            pts_2d = np.concatenate(pts_2d, axis=0)
            pts_2d_all[frame_id] = pts_2d
                                                
        np.savez_compressed(f'{seq_save_dir}/pointcloud.npz', 
                            pointcloud=pts_3d_all, 
                            camera_projection=pts_2d_all)
        print("Processing LiDAR data done...")

    if 'track' in process_list:
        print("Processing tracking data...")
        track_dir = os.path.join(seq_save_dir, "track")
        os.makedirs(track_dir, exist_ok=True)
        
        # Use GT tracker
        track_infos_path = os.path.join(track_dir, "track_info.txt")
        track_infos_file = open(track_infos_path, 'w')
        row_info_title = "frame_id " + "track_id " + "object_class " + "alpha " + \
                "box_height " + "box_width " + "box_length " + "box_center_x " + "box_center_y " + "box_center_z " \
                + "box_heading " \
                + "speed" + "\n"

        track_infos_file.write(row_info_title)
        track_vis_imgs = []
        bbox_visible_dict = dict()
        object_ids = dict()

        datafile = WaymoDataFileReader(seq_path)

        for frame_id, frame in tqdm(enumerate(datafile)):
            images = dict()
            for camera_name in camera_names_dict.keys():
                camera = utils.get(frame.images, camera_name)
                image = utils.decode_image(camera)
                images[camera_name] = image
            
            for label in frame.laser_labels:
                box = label.box
                speed = np.linalg.norm([label.metadata.speed_x, label.metadata.speed_y])

                # thresholding, use 1.0 m/s to determine whether the pixel is moving
                # follow EmerNeRF
                # if speed < 1.:
                #     continue
                
                # build 3D bounding box dimension
                length, width, height = box.length, box.width, box.height
                
                # build 3D bounding box pose
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c = math.cos(heading)
                s = math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                if label.id not in object_ids:
                    object_ids[label.id] = len(object_ids)

                label_id = object_ids[label.id]
                if label_id not in bbox_visible_dict:
                    bbox_visible_dict[label_id] = dict()
                if label_id not in all_obj_transformers:
                    all_obj_transformers[label_id] = np.array([[-1.,-1.,-1.]]*num_frames)
                if label_id not in all_obj_rotation_matrixes:
                    all_obj_rotation_matrixes[label_id] = np.array([np.eye(3).astype(np.float64)]*num_frames)
                bbox_visible_dict[label_id][frame_id] = []

                for camera_name in camera_names_dict.keys():
                    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                
                    vertices, valid = project_label_to_image(
                        dim=[length, width, height],
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    
                    # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                    # partial visible for the case when not all corners can be observed
                    if valid.any():
                        # print(f'At frame {frame_id}, label {label_id} is visible on {camera_names_dict[camera_name]}')
                        bbox_visible_dict[label_id][frame_id].append(camera_name-1)
                    if valid.all():
                        vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                        draw_3d_box_on_img(vertices, images[camera_name])
                    
                bbox_visible_dict[label_id][frame_id] = sorted(bbox_visible_dict[label_id][frame_id])
                    
                # assume every bbox is visible in at least on camera
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    obj_class = "vehicle"
                elif label.type == label_pb2.Label.Type.TYPE_PEDESTRIAN:
                    obj_class = "pedestrian"
                elif label.type == label_pb2.Label.Type.TYPE_SIGN:
                    obj_class = "sign"
                elif label.type == label_pb2.Label.Type.TYPE_CYCLIST:
                    obj_class = "cyclist"
                else:
                    obj_class = "misc"

                alpha = -10
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y])  
                lines_info = f"{frame_id} {label_id} {obj_class} {alpha} {height} {width} {length} {tx} {ty} {tz} {heading} {speed} \n"
                all_obj_transformers[label_id][frame_id] = np.array([tx,ty,tz])
                all_obj_rotation_matrixes[label_id][frame_id] = rotz_matrix
                track_infos_file.write(lines_info)
                
            track_vis_img = np.concatenate([
                images[dataset_pb2.CameraName.FRONT_LEFT], 
                images[dataset_pb2.CameraName.FRONT], 
                images[dataset_pb2.CameraName.FRONT_RIGHT]], axis=1)
            track_vis_imgs.append(track_vis_img)
        
        # save track visualization        
        # imageio.mimwrite(os.path.join(track_dir, "track_vis.mp4"), track_vis_imgs, fps=24)
        
        # save bbox visibility
        bbox_visible_path = os.path.join(track_dir, "track_camera_vis.json")
        with open(bbox_visible_path, 'w') as f:
            json.dump(bbox_visible_dict, f, indent=1)
            
        # save object ids mapping
        object_ids_path = os.path.join(track_dir, "track_ids.json")
        with open(object_ids_path, 'w') as f:
            json.dump(object_ids, f, indent=2)
        
        track_infos_file.close()

        # Use castrack
        if seq_name in castrack_infos:     
            track_infos_path = os.path.join(track_dir, "track_info_castrack.txt")
            track_infos_file = open(track_infos_path, 'w')
            row_info_title = "frame_id " + "track_id " + "object_class " + "alpha " + \
                "box_height " + "box_width " + "box_length " + "box_center_x " + "box_center_y " + "box_center_z " \
                + "box_heading " + "\n"
            track_infos_file.write(row_info_title)

            track_info = castrack_infos[seq_name]
            track_vis_imgs = []
            bbox_visible_dict = dict()
            object_ids = dict()

            datafile = WaymoDataFileReader(seq_path)
            for frame_id, frame in tqdm(enumerate(datafile)):
                images = dict()
                for camera_name in camera_names_dict.keys():
                    camera = utils.get(frame.images, camera_name)
                    image = utils.decode_image(camera)
                    images[camera_name] = image
                
                label = track_info[str(frame_id)]
                for i, object_id in enumerate(label['obj_ids']):
                    box = label['boxes_lidar'][i]

                    # build 3D bounding box dimension
                    length, width, height = box[3], box[4], box[5]
                    
                    # build 3D bounding box pose
                    tx, ty, tz = box[0], box[1], box[2]
                    heading = box[-1]
                    c = math.cos(heading)
                    s = math.sin(heading)
                    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    obj_pose_vehicle = np.eye(4)
                    obj_pose_vehicle[:3, :3] = rotz_matrix
                    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
                    
                    if object_id not in object_ids:
                        object_ids[object_id] = len(object_ids)
                        
                    label_id = object_ids[object_id]
                    if label_id not in bbox_visible_dict:
                        bbox_visible_dict[label_id] = dict()
                    bbox_visible_dict[label_id][frame_id] = []
                    
                    for camera_name in camera_names_dict.keys():
                        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                        
                        vertices, valid = project_label_to_image(
                            dim=[length, width, height],
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                        
                        # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
                        # partial visible for the case when not all corners can be observed
                        if valid.any():
                            # print(f'At frame {frame_id}, label {label_id} is visible on {camera_names_dict[camera_name]}')
                            bbox_visible_dict[label_id][frame_id].append(camera_name-1)
                        if valid.all():
                            vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
                            draw_3d_box_on_img(vertices, images[camera_name])
                      
                    bbox_visible_dict[label_id][frame_id] = sorted(bbox_visible_dict[label_id][frame_id])

                    # assume every bbox is visible in at least one camera
                    name = label['name'][i]
                    if name == 'Vehicle':
                        obj_class = "vehicle"
                    elif name == 'Cyclist':
                        obj_class = "cyclist"
                    elif name == 'Pedestrian':
                        obj_class = "pedestrian"
                    else:
                        obj_class = "misc"

                    alpha = -10

                    lines_info = f"{frame_id} {label_id} {obj_class} {alpha} {height} {width} {length} {tx} {ty} {tz} {heading} \n"
                    
                    track_infos_file.write(lines_info)
                
                track_vis_img = np.concatenate([
                    images[dataset_pb2.CameraName.FRONT_LEFT], 
                    images[dataset_pb2.CameraName.FRONT], 
                    images[dataset_pb2.CameraName.FRONT_RIGHT]], axis=1)
                track_vis_imgs.append(track_vis_img)
            
            # save track visualization
            # imageio.mimwrite(os.path.join(track_dir, "track_vis_castrack.mp4"), track_vis_imgs, fps=24)

            # save bbox visibility
            bbox_visible_path = os.path.join(track_dir, "track_camera_vis_castrack.json")
            with open(bbox_visible_path, 'w') as f:
                json.dump(bbox_visible_dict, f, indent=1)

            # save object ids mapping
            object_ids_path = os.path.join(track_dir, "track_ids_castrack.json")
            with open(object_ids_path, 'w') as f:
                json.dump(object_ids, f, indent=2)

            track_infos_file.close()
            print("Processing tracking data done...")

    if 'dynamic_mask' in process_list:
        print("Saving dynamic mask ...")
        dynamic_mask_dir = os.path.join(seq_save_dir, "dynamic_mask")
        os.makedirs(dynamic_mask_dir, exist_ok=True)
        datafile = WaymoDataFileReader(seq_path)

        for frame_id, frame in tqdm(enumerate(datafile)):
            masks = dict()
            for camera_name in camera_names_dict.keys():
                camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                width, height = camera_calibration.width, camera_calibration.height
                mask = np.zeros((height, width), dtype=np.uint8)
                masks[camera_name] = mask
    
            for label in frame.laser_labels:
                box = label.box
                meta = label.metadata
                speed = np.linalg.norm([meta.speed_x, meta.speed_y]) 
                
                # thresholding, use 1.0 m/s to determine whether the pixel is moving
                # follow EmerNeRF
                if speed < 1.:
                    continue
                
                # build 3D bounding box dimension
                length, width, height = box.length, box.width, box.height
                
                # build 3D bounding box pose
                tx, ty, tz = box.center_x, box.center_y, box.center_z
                heading = box.heading
                c = math.cos(heading)
                s = math.sin(heading)
                rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = rotz_matrix
                obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])

                for camera_name in camera_names_dict.keys():
                    camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
                    # dim = [length * 1.5, width * 1.5, height]
                    dim = [length, width, height]
                    vertices, valid = project_label_to_image(
                        dim=dim,
                        obj_pose=obj_pose_vehicle,
                        calibration=camera_calibration,
                    )
                    if valid.any():
                        mask = project_label_to_mask(
                            dim=dim,
                            obj_pose=obj_pose_vehicle,
                            calibration=camera_calibration,
                        )
                        masks[camera_name] = np.logical_or(
                            masks[camera_name], mask)
            
            for camera_name in camera_names_dict.keys():
                mask = masks[camera_name]
                mask_path = os.path.join(dynamic_mask_dir, f'{frame_id:06d}_{str(camera_name - 1)}.png')
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

        print("Saving dynamic mask done...")

    return cam_extrinsics, all_obj_transformers, all_obj_rotation_matrixes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_list', type=str, nargs='+', default=['pose', 'calib', 'image', 'lidar', 'track', 'dynamic_mask'])
    parser.add_argument('--root_dir', type=str, default='/nas/home/yanyunzhi/waymo/training')
    parser.add_argument('--save_dir', type=str, default='/nas/home/yanyunzhi/waymo/street_gaussian/training/surrounding')
    parser.add_argument('--track_file', type=str, default='/nas/home/yanyunzhi/waymo/castrack/seq_infos/val/result.json')
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--segment_file', type=str)
    args = parser.parse_args()
    
    process_list = args.process_list
    root_dir = args.root_dir
    save_dir = args.save_dir
    track_file = args.track_file
    split_file = open(args.split_file, "r").readlines()[1:]
    scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    seq_names = [line.strip().split(",")[1] for line in split_file]
    segment_file = args.segment_file

    seq_lists = open(segment_file).read().splitlines()
    # seq_lists = open(os.path.join(root_dir, 'segment_list.txt')).read().splitlines()
    os.makedirs(save_dir, exist_ok=True)
    for i, scene_id in enumerate(scene_ids_list):
        assert seq_names[i][3:] == seq_lists[scene_id][8:14]
        seq_save_dir = os.path.join(save_dir, str(scene_id).zfill(3))
        parse_seq_rawdata(
            process_list=process_list,
            root_dir=root_dir,
            seq_name=seq_lists[scene_id],
            seq_save_dir=seq_save_dir,
            track_file=track_file,
        )
    
if __name__ == '__main__':
    # main()
    raw_dir = r'/home/anchun/src/hierarchical-3d-gaussians/data/notr/raw'
    output_dir = r'/home/anchun/src/hierarchical-3d-gaussians/data/notr/processed/notr_026'
    # raw_dir = r'E:\data\hierarchical_3dgs\notr/raw'
    # output_dir = r'E:\data\hierarchical_3dgs\notr/processed/notr_026'
    # raw_dir = r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\raw'
    # output_dir = r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\processed/notr_026'
    # 第一步，把原始waymo转换为notr格式
    cam_extrinsics, all_obj_transformers, all_obj_rotation_matrixes = parse_seq_rawdata(
        process_list=['pose', 'calib', 'image', 'track', 'dynamic_mask'], # 'lidar'
        root_dir=raw_dir,
        seq_name='segment-12374656037744638388_1412_711_1432_711_with_camera_labels',
        seq_save_dir=output_dir,
        track_file=output_dir + '/object_infos.txt',
    )

    # 第二步，把notr格式转换成colmap格式，需要调用colmap，并收集场景信息
    scene_infos = generate_dataparser_outputs(output_dir,build_pointcloud=False)
    scene_infos['extrinsics'] = cam_extrinsics
    del scene_infos['exts']
    del scene_infos['ixts']
    del scene_infos['poses']
    del scene_infos['c2ws']
    del scene_infos['obj_tracklets']
    del scene_infos['frames']
    del scene_infos['cams']
    del scene_infos['frames_idx']
    del scene_infos['obj_bounds']
    for static_obj_id in scene_infos['static_object_ids']:
        if static_obj_id in all_obj_transformers.keys():
            del all_obj_transformers[static_obj_id]
        if static_obj_id in all_obj_rotation_matrixes.keys():
            del all_obj_rotation_matrixes[static_obj_id]
        if static_obj_id in scene_infos['obj_info'].keys():
            del scene_infos['obj_info'][static_obj_id]
    for obj_id in scene_infos['obj_info'].keys():
        start_frame = scene_infos['obj_info'][obj_id]['start_frame']
        end_frame = scene_infos['obj_info'][obj_id]['end_frame']
        if obj_id in all_obj_transformers.keys():
            all_obj_transformers[obj_id][0:start_frame] = np.array([-1.,-1.,-1.])
            all_obj_transformers[obj_id][end_frame+1:] = np.array([-1., -1., -1.])
        if obj_id in all_obj_rotation_matrixes.keys():
            all_obj_rotation_matrixes[obj_id][0:start_frame] = np.eye(3).astype(np.float64)
            all_obj_rotation_matrixes[obj_id][end_frame+1:] = np.eye(3).astype(np.float64)
    scene_infos['all_obj_transformers'] = all_obj_transformers
    scene_infos['all_obj_rotation_matrixes'] = all_obj_rotation_matrixes

    joblib.dump(scene_infos, os.path.join(output_dir, 'scene_meta.bin'))
