import numpy as np
import cv2
import os
import sys
import time

import torch
from torch import nn

from models.yolo import *
from models.hrnet import *
from utils.detector import *


def predict(file_path, pred_path, module_dir, draw_bbox=False, box_tr=0.7):
    # file_path - absolute path to file
    # pred_path - absolute path for prediction
    # module_dir - path for module folder
    # draw_bbox - draw bboxes or not
    # box_tr - threshold for bbox confidence

    image_formats = ['.jpg', '.png', '.jpeg', '.bmp']
    video_formats = ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v']
    file_format = file_path[file_path.rindex('.'):].lower()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    yolov5 = load_yolo_model(module_dir).to(device)
    keypoint_net = load_keypoint_net(module_dir).to(device)

    if file_format in image_formats:
        pred_path = predict_image(file_path, pred_path, yolov5, keypoint_net, device,
                                  draw_bbox=draw_bbox, box_tr=box_tr)

    elif file_format in video_formats:
        pred_path = predict_video(file_path, pred_path, yolov5, keypoint_net, device,
                                  draw_bbox=draw_bbox, box_tr=box_tr)

    else:
        print('Unknown file format')

    return pred_path
