import numpy as np
import argparse
import os, time
from read_write_model import *
from scipy.spatial.transform import Rotation

def rotate_camera(qvec, tvec, rot_matrix):
    # Assuming cameras have 'T' (translation) field

    R = qvec2rotmat(qvec)
    T = np.array(tvec)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = np.copy(C2W[:3, 3])
    cam_rot_orig = np.copy(C2W[:3, :3])
    cam_center = np.matmul(cam_center, rot_matrix)
    cam_rot = np.linalg.inv(rot_matrix) @ cam_rot_orig
    C2W[:3, 3] = cam_center
    C2W[:3, :3] = cam_rot
    Rt = np.linalg.inv(C2W)
    new_pos = Rt[:3, 3]
    new_rot = rotmat2qvec(Rt[:3, :3])

    # R_test = qvec2rotmat(new_rots[-1])
    # T_test = np.array(new_poss[-1])
    # Rttest = np.zeros((4, 4))
    # Rttest[:3, :3] = R_test
    # Rttest[:3, 3] = T_test
    # Rttest[3, 3] = 1.0
    # C2Wtest = np.linalg.inv(Rttest) 

    return new_pos, new_rot

def parse_angle(s):
    s = s.strip("[]")
    return [float(x) for x in s.split(",")]

def main():

    parser = argparse.ArgumentParser(description='Example script with command-line arguments.')
    
    # Add command-line argument(s)
    parser.add_argument('--input_path', type=str, help='Path to input colmap dir',  required=True)
    parser.add_argument('--output_path', type=str, help='Path to output colmap dir',  required=True)
    parser.add_argument('--angle', type=parse_angle, help="input for rotation angles in zyx order [rotz, roty,rotx] in degrees", default="[0,90,90]")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    global_start = time.time()

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments    
    os.makedirs(args.output_path, exist_ok=True)

    # Your main logic goes here
    print("Input path:", args.input_path)
    print("Output path:", args.output_path)

    rotation_matrix = Rotation.from_euler('zyx', args.angle, degrees=True).as_matrix()
    rotation_matrix = np.linalg.inv(rotation_matrix)

    # Read colmap cameras, images and points
    start_time = time.time()
    cameras, images_metas_in, points3d_in = read_model(args.input_path)
    end_time = time.time()
    print(f"{len(images_metas_in)} images read in {end_time - start_time} seconds.")

    positions = []
    print("Doing points")
    for key in points3d_in: 
        positions.append(points3d_in[key].xyz)
    
    positions = np.array(positions)
    
    # Perform the rotation by matrix multiplication
    rotated_points = np.matmul(positions, rotation_matrix)

    points3d_out = {}
    for key, rotated in zip(points3d_in, rotated_points):
        point3d_in = points3d_in[key]
        points3d_out[key] = Point3D(
            id=point3d_in.id,
            xyz=rotated,
            rgb=point3d_in.rgb,
            error=point3d_in.error,
            image_ids=point3d_in.image_ids,
            point2D_idxs=point3d_in.point2D_idxs,
        )

    print("Doing images")
    images_metas_out = {} 
    for key in images_metas_in: 
        image_meta_in = images_metas_in[key]
        new_pos, new_rot = rotate_camera(image_meta_in.qvec, image_meta_in.tvec, rotation_matrix)
        
        images_metas_out[key] = Image(
            id=image_meta_in.id,
            qvec=new_rot,
            tvec=new_pos,
            camera_id=image_meta_in.camera_id,
            name=image_meta_in.name,
            xys=image_meta_in.xys,
            point3D_ids=image_meta_in.point3D_ids,
        )

    write_model(cameras, images_metas_out, points3d_out, args.output_path)

    global_end = time.time()

    print(f"reorient script took {global_end - global_start} seconds.")

if __name__ == "__main__":
    main()