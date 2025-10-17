from predictor import VisualizationDemo
from detectron2.utils.logger import setup_logger
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from pathlib import Path
from skimage import io, morphology
import tqdm
import numpy as np
import cv2
import argparse
import os
import sys

mask2former_path = os.path.join(Path.home(), "src/Mask2Former")
sys.path.append(mask2former_path)
from mask2former import add_maskformer2_config


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def label2mask(label):
    # Bird, Ground Animal, 2*Curb, Fence, Guard Rail,
    # Barrier, Wall, 7*Bike Lane, 8*Crosswalk - Plain, 9*Curb Cut,
    # Parking, Pedestrian Area, Rail Track, 13*Road, 14*Service Lane,
    # **15*Sidewalk, Bridge, Building, Tunnel, Person,
    # Bicyclist, Motorcyclist, Other Rider, 23*Lane Marking - Crosswalk, 24*Lane Marking - General,
    # Mountain, Sand, Sky, Snow, **29*Terrain,
    # Vegetation, Water, Banner, Bench, Bike Rack,
    # Billboard, Catch Basin, CCTV Camera, Fire Hydrant, Junction Box,
    # Mailbox, 41*Manhole, Phone Booth, Pothole, Street Light,
    # Pole, Traffic Sign Frame, Utility Pole, Traffic Light, Traffic Sign (Back),
    # Traffic Sign (Front), Trash Can, Bicycle, Boat, Bus,
    # Car, Caravan, Motorcycle, On Rails, Other Vehicle,
    # Trailer, Truck, Wheeled Slow, Car Mount, Ego Vehicle
    mask = np.ones_like(label)
    # label_off_road = ((0 <= label) & (label <= 1)) | ((3 <= label) & (label <= 6)) | ((10 <= label) & (label <= 12)) \
    #                  | ((16 <= label) & (label <= 22)) | ((25 <= label) & (label <= 28)) | (
    #                          (30 <= label) & (label <= 40)) | (label >= 42)
    label_off_road = (label != 2) & (label != 7) & (label != 8) & (label != 9) & (label != 13) & (label != 14) & (label != 23) & (label != 24) & (label != 41)

    # dilate itereation 2 for moving objects
    label_movable = label >= 52
    kernel = np.ones((10, 10), dtype=np.uint8)
    label_movable = cv2.dilate(label_movable.astype(np.uint8), kernel, 2).astype(bool)

    label_off_road = label_off_road | label_movable
    mask[label_off_road] = 0
    label[~(mask.astype(bool))] = 64
    return mask, label

def clean_binary_mask(mask, min_obj_size=1000):
    # 确保转换为 bool 类型
    mask_bool = mask > 0
    # 去除小白块（小的前景区域）
    cleaned = morphology.remove_small_objects(mask_bool, min_size=min_obj_size)
    # 暂时不填补黑洞，确保路面语义正确
    #cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_obj_size)
    # 转为 0/1 uint8 格式输出
    cleaned_mask = (cleaned.astype(np.uint8))
    return cleaned_mask

def get_parser():
    parser = argparse.ArgumentParser(description="road segmentation using Mask2Former")
    parser.add_argument('--project_dir', required=True, help="project directory")

    parser.add_argument(
        "--config-file",
        default=os.path.join(mask2former_path, "configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml"),
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", os.path.join(mask2former_path, "mask2former_mapillary_vistas_swin_L.pkl")],
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = get_parser()
    images_dir = os.path.join(args.project_dir, "camera_calibration/rectified/images")
    output_roadmask_dir = os.path.join(args.project_dir, "camera_calibration/rectified/roadmasks")
    
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    if not os.path.isdir(images_dir):
        raise ValueError(f"images_dir {images_dir} is not a valid directory")

    demo = VisualizationDemo(cfg)
    
    
    image_root = Path(images_dir)
    extensions=('jpg', 'jpeg', 'png')
    file_paths = [str(p) for p in image_root.rglob('*') if p.suffix.lower().lstrip('.') in extensions]
    # for testing
    # file_paths = [os.path.join(images_dir, 'left-backward/1702406286683305.jpeg')]

    save_paths = [file_path.replace(images_dir, output_roadmask_dir)  for file_path in file_paths]
    roadmask_save_paths = [Path(save_path).with_suffix('.png') for save_path in save_paths]
    segmentation_save_paths = [Path(save_path).with_suffix('.seg.png') for save_path in save_paths]
    
    for i in tqdm.tqdm(range(len(file_paths))):
        # use PIL, to be consistent with evaluation
        source_file_path = file_paths[i]
        roadmask_save_path = roadmask_save_paths[i]
        roadmask_save_path.parent.mkdir(parents=True, exist_ok=True)
        img = read_image(source_file_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        label_image = predictions["sem_seg"].argmax(dim=0).cpu().numpy()  # (H, W)
        #cv2.imwrite(str(segmentation_save_paths[i]), label_image)
        road_mask, _ = label2mask(label_image)
        cleaned_road_mask = clean_binary_mask(road_mask, min_obj_size=2500)
        cv2.imwrite(str(roadmask_save_path), cleaned_road_mask * 255)
