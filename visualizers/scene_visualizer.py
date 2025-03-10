import os
import torchvision
import cv2
import torch
import imageio
import numpy as np

from utils.camera_utils import Camera
from utils.image_utils import visualize_depth_numpy


class SceneVisualizer():
    def __init__(self, save_dir):
        self.result_dir = save_dir
        os.makedirs(self.result_dir, exist_ok=True)

        self.save_video = True
        self.fps = 10

    def make_groups(self, group_names):
        self.groups = {name: [] for name in group_names}

    def append_one_frame(self, frame_groups):
        for name, img in frame_groups.items():
            self.groups[name].append(img)

    def merge_rendered_groups_as_one_mp4(self, output_path, sorted_names, layout):
        if self.save_video and len(sorted_names) > 0:
            # 获取最大的行数和列数
            max_rows = max([pos[0] for pos in layout.values()]) + 1
            max_cols = max([pos[1] for pos in layout.values()]) + 1

            combined_frames = []
            num_frames = len(self.groups[sorted_names[0]])

            for i in range(num_frames):
                # 初始化一个空白的画布
                canvas = None
                row_canvases = []
                for row in range(max_rows):
                    col_canvases = []
                    for col in range(max_cols):
                        for name in sorted_names:
                            if layout.get(name) == (row, col):
                                frame = self.groups[name][i]
                                if canvas is None:
                                    # 初始化画布的高度和宽度
                                    canvas_height = max_rows * frame.shape[0]
                                    canvas_width = max_cols * frame.shape[1]
                                    canvas = np.zeros((canvas_height, canvas_width, frame.shape[2]), dtype=frame.dtype)
                                col_canvases.append(frame)
                                break
                        else:
                            # 如果该位置没有指定视频，添加一个空白帧
                            if canvas is not None:
                                col_canvases.append(np.zeros_like(self.groups[sorted_names[0]][i]))
                    # 水平拼接列
                    row_canvas = np.hstack(col_canvases)
                    row_canvases.append(row_canvas)
                # 垂直拼接行
                combined_frame = np.vstack(row_canvases)
                combined_frames.append(combined_frame)

            imageio.mimwrite(output_path, combined_frames, fps=self.fps)

