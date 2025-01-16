from tqdm import tqdm
import math
import random
from PIL import Image
import os
import sys
import io
sys.path.append('./')
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.utils.frame_utils import *
from scipy.spatial.transform import Rotation as R
import joblib


# camera_names = ['FRONT','FRONT_LEFT','FRONT_RIGHT','SIDE_LEFT', 'SIDE_RIGHT']
# camera_names = ['FRONT','FRONT_LEFT','FRONT_RIGHT']
# camera_ids = {name: idx+1 for idx, name in enumerate(camera_names)}


# opencv2camera = np.array([[0., 0., 1., 0.],
#                         [-1., 0., 0., 0.],
#                         [0., -1., 0., 0.],
#                         [0., 0., 0., 1.]])
waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}


def get_extrinsic(camera_calibration):
    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4) # camera to vehicle
    # extrinsic = np.matmul(camera_extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
    return extrinsic


def get_intrinsic(camera_calibration):
    camera_intrinsic = camera_calibration.intrinsic
    fx = camera_intrinsic[0]
    fy = camera_intrinsic[1]
    cx = camera_intrinsic[2]
    cy = camera_intrinsic[3]
    intrinsic = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])
    return intrinsic


camera_names_dict = {
    dataset_pb2.CameraName.FRONT_LEFT: 'FRONT_LEFT',
    dataset_pb2.CameraName.FRONT_RIGHT: 'FRONT_RIGHT',
    dataset_pb2.CameraName.FRONT: 'FRONT',
    dataset_pb2.CameraName.SIDE_LEFT: 'SIDE_LEFT',
    dataset_pb2.CameraName.SIDE_RIGHT: 'SIDE_RIGHT',
}
camera_ids = {name: id for id, name in camera_names_dict.items()}

def generate_image_name(frame_id, camera_name):
    return '{}_{}.jpg'.format(camera_name.lower(), frame_id)


def extract_images(frame_id, image_dir, camera_name, image_array):
    image_array = np.array(Image.open(io.BytesIO(image_array)))
    image = Image.fromarray(image_array)
    img_name = generate_image_name(frame_id, camera_name)
    image.save(os.path.join(image_dir, img_name))


def matrix_to_euler_zyx(matrix):
    # 提取旋转矩阵部分
    rotation_matrix = matrix[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler('zyx', False)
    translation_vector = matrix[:3, 3]
    euler_angles = [euler_angles[2], euler_angles[1], euler_angles[0]]
    return euler_angles, translation_vector


def write_camera_pose_to_colmap(frame_id, camera_name, ego_pose_in_world, camera_pose_in_ego, output_file):
    # 第一步：把相机位姿从主车坐标系转换成世界坐标系
    camera_pose_in_world = np.dot(ego_pose_in_world, camera_pose_in_ego)

    # 把相机位姿从世界坐标系转换成世界原点在colmap相机坐标系中的位姿:
    rot, pos = matrix_to_euler_zyx(camera_pose_in_world)
    rot2camera = R.from_euler('yx', [90, -90], degrees=True).as_matrix() # TODO: 这一步缺解释
    rotationInverse = np.linalg.inv(R.from_euler('xyz', rot, False).as_matrix().dot(rot2camera))  #
    # rotationInverse = np.linalg.inv(R.from_euler('xyz', rot, False).as_matrix())
    # pos = -rotationInverse.as_matrix().dot(pos)
    pos = -rotationInverse.dot(pos)
    q = R.from_matrix(rotationInverse).as_quat()
    w, x, y, z = q[3], q[0], q[1], q[2]

    img_name = generate_image_name(frame_id, camera_name)
    img_id = frame_id*len(camera_names_dict) + camera_ids[camera_name]
    output = '{} {} {} {} {} {} {} {} {} {} \n'.format(img_id,
                                                  w,x,y,z,
                                                  pos[0], pos[1], pos[2],
                                                  camera_ids[camera_name], img_name)
    output_file.write(output + '\n')
    return img_id


def extract_points(frame, frame_data_detail, point_cloud, offset):
    class SimoneCloudPoint:
        x, y, z = 0.0, 0.0, 0.0
        argb = 0
        r,g,b = 0, 0, 0
        intensity = 0
        segmentation = 0
        ring = 0
        angle = 0

        def __hash__(self):
            return hash((self.x, self.y, self.z, self.argb, self.intensity, self.segmentation, self.ring, self.angle))

        def __eq__(self, other):
            if isinstance(other, SimoneCloudPoint):
                return (self.x == other.x and
                        self.y == other.y and
                        self.z == other.z and
                        self.argb == other.argb and
                        self.r == other.r and
                        self.g == other.g and
                        self.b == other.b and
                        self.intensity == other.intensity and
                        self.segmentation == other.segmentation and
                        self.ring == other.ring and
                        self.angle == other.angle)
            return False
    range_images, camera_projections, _, range_image_top_pose = parse_range_image_and_camera_projection(frame)
    points, _ = convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
    lidar_pose_in_ego = frame_data_detail['FRONT_LIDAR_EXTRINSIC']
    ego_pose_in_world = frame_data_detail['FRONT_POSE']
    lidar_pos_in_world = np.dot(ego_pose_in_world, lidar_pose_in_ego)

    points = np.concatenate([points[0], np.ones((len(points[0]), 1))], axis=1)
    points = np.dot(lidar_pos_in_world, points.T).T
    points = points[:, :3] - offset
    # points = np.random.choice(len(points), len(points)//4, replace=False)
    for i,point in enumerate(points):
        if random.randint(0,100)<8:
            p = SimoneCloudPoint()
            # p.x, p.y, p.z = -point[1], -point[2], point[0]
            p.x, p.y, p.z = point[0], point[1], point[2]
            point_cloud.add(p)

def bbox_to_corner3d(bbox):
    min_x, min_y, min_z = bbox[0]
    max_x, max_y, max_z = bbox[1]

    corner3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corner3d

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


def parse_dynamic_objects(frame_id, frame, dynamic_objects):
    bbox_visible_dict = dict()
    object_ids = dict()
    frame_objects = []
    for label in frame.laser_labels:
        box = label.box
        length, width, height = box.length, box.width, box.height
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
        bbox_visible_dict[label_id][frame_id] = []

        for camera_name in camera_names_dict.keys():
            camera_calibration = get_content_from_list(frame.context.camera_calibrations, camera_name)

            vertices, valid = project_label_to_image(
                dim=[length, width, height],
                obj_pose=obj_pose_vehicle,
                calibration=camera_calibration,
            )

            # if one corner of the 3D bounding box is on camera plane, we should consider it as visible
            # partial visible for the case when not all corners can be observed
            if valid.any():
                # print(f'At frame {frame_id}, label {label_id} is visible on {camera_names_dict[camera_name]}')
                bbox_visible_dict[label_id][frame_id].append(camera_name - 1)
            # if valid.all():
            #     vertices = vertices.reshape(2, 2, 2, 2).astype(np.int32)
            #     draw_3d_box_on_img(vertices, images[camera_name])

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

        if obj_class == 'pedestrian':
            deformable = True
        else:
            deformable = False
        meta = label.metadata
        speed = np.linalg.norm([meta.speed_x, meta.speed_y])
        frame_objects.append({'track_id': label_id, 'class': obj_class, 'class_label': waymo_track2label[obj_class], 'deformable': deformable
                                 , 'width': width, 'height': height, 'length': length
                                 , 'transform': np.array([tx,ty,tz]), 'rotation_matrix': rotz_matrix, 'speed': speed})
    dynamic_objects.append(frame_objects)


def get_content_from_list(object_list, name):
    """ Search for an object by name in an object list. """

    object_list = [obj for obj in object_list if obj.name == name]
    return object_list[0]


# 以 object_id组织动态对象，每个动态对象包括：
#    基础元数据：如长宽高、类别等
#    起始帧号：在任一一个视角出现的最开始和最后一帧
#    位姿：key为帧号，value为transformer和rotation(3*3矩阵)
def convert_and_filter(dynamic_objects):
    output = {}
    for frame_id, frame_objects in enumerate(dynamic_objects):
        for obj in frame_objects:
            track_id = obj['track_id']
            if track_id not in output:
                output[track_id] = {'object_id': track_id, 'start_frame':1e10, 'end_frame':0, 'deformable': obj['deformable'], 'class': obj['class'], 'class_label': obj['class_label'], 'width': obj['width'], 'height': obj['height'], 'length': obj['length'], 'all_transforms':{}, 'all_rotation_matrixs':{}}
            output[track_id]['start_frame'] = min(output[track_id]['start_frame'], frame_id)
            output[track_id]['end_frame'] = max(output[track_id]['end_frame'], frame_id)
            output[track_id]['all_transforms'][str(frame_id)] = obj['transform']
            output[track_id]['all_rotation_matrixs'][str(frame_id)] = obj['rotation_matrix']
    return output


def extract_waymo_scene(input_path, output_dir):
    image_dir = os.path.join(output_dir, 'raw_images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    colmap_sparse_model_dir = os.path.join(output_dir, 'sparse/0')
    if not os.path.exists(colmap_sparse_model_dir):
        os.makedirs(colmap_sparse_model_dir)
    dataset = tf.data.TFRecordDataset(input_path)
    colmap_images_file = open(os.path.join(colmap_sparse_model_dir, "images.txt"), 'w')
    first_frame_pos = None
    timestamps = []
    ego_poses = []
    camera_intrinsics = {}
    img_id_2_frame_id = {} # colmap 的images.txt中以image id标识每个视图，因为加入了动态性，需要把每个视图关联到对应的帧号
    img_id_2_extrinsic = {} # colmap 的images.txt中以image id标识每个视图，因为加入了动态性，需要把每个视图关联到对应的外参
    dynamic_objects = []
    point_cloud = set()
    num_frames = 0
    for frame_id, raw_frame in tqdm(enumerate(dataset), total=199):
        frame_image_names = {}
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(raw_frame.numpy()))
        ego_pose = np.array(frame.pose.transform).reshape(4, 4)
        if first_frame_pos is None:
            first_frame_pos = ego_pose[:3,3].copy()
        ego_poses.append(ego_pose)
        ego_pose[:3, 3] = ego_pose[:3, 3] - first_frame_pos
        timestamps.append(frame.timestamp_micros / 1e6)
        for i, camera in enumerate(frame.context.camera_calibrations):
            camera_name = camera_names_dict[camera.name]
            more_camera_detail = get_content_from_list(frame.images, camera.name)
            camera_timestamp = more_camera_detail.pose_timestamp
            extract_images(frame_id, image_dir, camera_name, more_camera_detail.image)
            intrinsic = camera.intrinsic
            camera_intrinsics[camera_name] = ['PINHOLE', camera.width, camera.height, intrinsic[0], intrinsic[1], intrinsic[2],intrinsic[3]]
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            # extrinsic = np.matmul(extrinsic, opencv2camera) # [forward, left, up] to [right, down, forward]
            camera_pose = np.array(more_camera_detail.pose.transform).reshape(4, 4)
            camera_pose[:3, 3] = camera_pose[:3, 3] - first_frame_pos
            frame_image_names[camera_name] = generate_image_name(frame_id, camera_name)
            image_id = write_camera_pose_to_colmap(frame_id, camera_name, ego_pose, extrinsic, colmap_images_file)
            img_id_2_frame_id[image_id] = frame_id
            img_id_2_extrinsic[image_id] = extrinsic
        parse_dynamic_objects(frame_id, frame, dynamic_objects)
        # extract_points(frame, convert_frame_to_dict(frame), point_cloud, first_frame_pos)
        num_frames = num_frames + 1
    colmap_images_file.flush()
    colmap_images_file.close()
    with open(os.path.join(colmap_sparse_model_dir, "cameras.txt"), 'w') as camera_intrinsic_file:
        for camera_name, intrinsic in camera_intrinsics.items():
            camera_id = camera_ids[camera_name]
            output = '{} {} {} {} {} {} {} {} \n'.format(camera_id, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], intrinsic[4], intrinsic[5], intrinsic[6])
            camera_intrinsic_file.write(output)

    # with open(colmap_sparse_model_dir + "/points3D.txt","w") as output:
    #     for i, p in enumerate(point_cloud):
    #         r=random.randint(0,255)
    #         g=random.randint(0,255)
    #         b=random.randint(0,255)
    #         output.write(" ".join([str(c) for c in [i + 1, p.x, p.y, p.z, r, g, b, 0.01, 1, 0, 2, 0, 3,0]]) + "\n")  # rgb后面的几个值是fake的，3DGS中没用。这里为了让colmap gui加载时不崩溃
    dynamic_objects = convert_and_filter(dynamic_objects)
    scene_meta = {'timestamps': timestamps, 'ego_poses': ego_poses, 'dynamic_objects': dynamic_objects, 'num_frames': num_frames,
                  'img_id_2_frame_id':img_id_2_frame_id, 'img_id_2_extrinsic': img_id_2_extrinsic}
    joblib.dump(scene_meta, os.path.join(output_dir, 'scene_meta.bin'))


if __name__ == '__main__':
    # tf.enable_eager_execution()
    input_path = r'D:\Projects\51sim-ai\EmerNeRF\data\waymo\raw/segment-10588771936253546636_2300_000_2320_000_with_camera_labels.tfrecord'
    out_dir = r'D:\Projects\51sim-ai\EmerNeRF\data\waymo/processed/031'
    os.makedirs(out_dir, exist_ok=True)
    extract_waymo_scene(input_path, out_dir)
