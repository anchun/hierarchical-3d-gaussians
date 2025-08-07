import os, sys, shutil
import posixpath
import numpy as np
from database import COLMAPDatabase
from scipy.spatial.transform import Rotation as R

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


def update_db_for_colmap_models(db, model_path, pose_prior_variance=0.25):
    # read camera info
    camera_ids = []
    cameras = []
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
            db.add_camera(camModelDict[camera_model], w, h, params, prior_focal_length=True, camera_id=len(camera_ids))
            camera_ids.append(camera_id)
            cameras.append([camera_id, w, h, fx, fy, cx, cy])
            camera_images[camera_id] = []

    # read image info.
    images_file = posixpath.join(model_path, "images.txt")

    count = 0
    with open(images_file, "r") as fid:
        for line in fid:
            if line.startswith('#'):
                continue
            values = line.strip().split(' ')
            if len(values) != 10:
                continue
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name = int(values[0]), float(values[1]), float(
                values[2]), float(values[3]), float(values[4]), float(values[5]), float(values[6]), float(
                values[7]), int(values[8]), values[9]
            # print(image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, image_name)
            if camera_id in camera_ids:
                camera_images[camera_id].append(
                    {'image_name': image_name, 'prior_q': [qw, qx, qy, qz], 'prior_t': [tx, ty, tz]})
                count += 1

    # add db and update image_id with new sequence
    count = 0
    for camera_idx, camera_id in enumerate(camera_ids):
        # sort by name first
        camera_images[camera_id].sort(key=lambda x: x['image_name'])
        # add images to db
        for camera_image in camera_images[camera_id]:
            camera_image['image_id'] = count
            db.add_image(camera_image['image_name'], camera_idx, image_id=camera_image['image_id'])
            R_mat = R.from_quat(camera_image['prior_q']).as_matrix()
            pose_world = -R_mat.T @ camera_image['prior_t']
            db.add_pose_prior(camera_image['image_id'], pose_world, 1, # "WGS84": 0, "CARTESIAN": 1
                              position_covariance = np.eye(3) * pose_prior_variance)
            count += 1

    # refine cameras.txt with new id
    if len(cameras) > 0:
        HEADER = '# # Camera list with one line of data per camera:\n' + \
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + \
                 "# Number of cameras: {}\n".format(len(cameras))
        with open(cameras_file, "w") as fid:
            fid.write(HEADER)
            for idx, camera in enumerate(cameras):
                camera_id, w, h, fx, fy, cx, cy = camera
                camera_header = [idx, "PINHOLE", w, h, fx, fy, cx, cy]
                first_line = " ".join(map(str, camera_header))
                fid.write(first_line + "\n")
                    
    # refine images.txt with new id
    if len(camera_images) > 0:
        HEADER = '# Images list with two lines of data per image:\n' + \
                 "#   Images_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" + \
                 "#   POINTS2D[] AS (X, Y, POINT3D_ID)\n" + \
                 "# Number of images: {}, mean obersevation per image: {}\n".format(count, 0)
        with open(images_file, "w") as fid:
            fid.write(HEADER)
            for idx in range(len(camera_images[camera_ids[0]])):
                for camera_idx, camera_id in enumerate(camera_ids):
                    camera_image = camera_images[camera_id][idx]
                    image_header = [camera_image['image_id'], camera_image['prior_q'][0], camera_image['prior_q'][1],
                                    camera_image['prior_q'][2], camera_image['prior_q'][3],
                                    camera_image['prior_t'][0], camera_image['prior_t'][1], camera_image['prior_t'][2],
                                    camera_idx, camera_image['image_name']]
                    first_line = " ".join(map(str, image_header))
                    fid.write(first_line + "\n")
                    fid.write("\n")