import numpy as np
import cv2
import os
import sys
import time

import torch
import torch.nn as nn
import torchvision.transforms as T

from utils.coordop import *
from utils.yolo_utils import prepare_image, yolo_predict


def predict_image(file_path, pred_path, yolov5, keypoint_net, device, draw_bbox=False, box_tr=0.5):
    # Function for keypoints prediction for a single image
    # pred_path = pred_path + '.jpg'
    img = cv2.imread(file_path)
    keypoint_detector = KeypointDetector(yolov5, keypoint_net, device, 4, draw_bbox, box_tr)
    p_img = keypoint_detector.detect_keypoints(img)
    cv2.imwrite(pred_path, np.uint8(p_img))

    return pred_path


def predict_video(file_path, pred_path, yolov5, keypoint_net, device, draw_bbox=False, box_tr=0.5):
    # Function for keypoints prediction for a video
    # pred_path = pred_path + '.webm'
    file_format = file_path[file_path.rindex('.'):].lower()

    if file_format == '.mov':
        rotate = True
    else:
        rotate = False
    # Videoreader
    video_reader = cv2.VideoCapture(file_path)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(frame_h, frame_w)

    # Setting videowriter
    if rotate:
        vw_w, vw_h = frame_h, frame_w
    else:
        vw_w, vw_h = frame_w, frame_h

    video_writer = cv2.VideoWriter(pred_path,
                                   cv2.VideoWriter_fourcc(*'VP80'),
                                   24.0,
                                   (vw_w, vw_h))

    images = []  # List for postpreccessed images
    keypoint_detector = KeypointDetector(yolov5, keypoint_net, device, 4, draw_bbox, box_tr)
    for i in range(nb_frames - 1):
        ok, img = video_reader.read()

        if ok:
            if rotate:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            p_img = keypoint_detector.detect_keypoints(img)
            images.append(p_img)

    for i in range(len(images)):
        video_writer.write(images[i].astype('uint8'))

    video_reader.release()
    video_writer.release()

    return pred_path


class KeypointDetector():
    # Yolo parameters
    name_classes = ['hand']
    num_classes = len(name_classes)
    input_shape = (416, 416)

    # Colors for keypoints
    point_colors = [[250, 0, 0],
                    [200, 0, 0],
                    [150, 0, 0],
                    [100, 0, 0],
                    [50, 0, 0],
                    [250, 250, 250],
                    [200, 200, 200],
                    [150, 150, 150],
                    [100, 100, 100],
                    [0, 250, 0],
                    [0, 200, 0],
                    [0, 150, 0],
                    [0, 100, 0],
                    [0, 250, 250],
                    [0, 200, 200],
                    [0, 150, 150],
                    [0, 100, 100],
                    [0, 0, 250],
                    [0, 0, 200],
                    [0, 0, 150],
                    [0, 0, 100]]

    # Colors for keypoint connecting lines
    line_colors = [[250, 0, 0],
                   [200, 0, 0],
                   [150, 0, 0],
                   [100, 0, 0],
                   [250, 0, 0],
                   [200, 200, 200],
                   [150, 150, 150],
                   [100, 100, 100],
                   [250, 0, 0],
                   [0, 200, 0],
                   [0, 150, 0],
                   [0, 100, 0],
                   [250, 0, 0],
                   [0, 200, 200],
                   [0, 150, 150],
                   [0, 100, 100],
                   [250, 0, 0],
                   [0, 0, 200],
                   [0, 0, 150],
                   [0, 0, 100]]

    # Color for bbox
    colors = [[255, 255, 255]]

    # Indices of keypoint pairs for connection
    connection_indices = [[0, 1], [1, 2], [2, 3], [3, 4],
                          [0, 5], [5, 6], [6, 7], [7, 8],
                          [0, 9], [9, 10], [10, 11], [11, 12],
                          [0, 13], [13, 14], [14, 15], [15, 16],
                          [0, 17], [17, 18], [18, 19], [19, 20]]

    # Hand image config
    IMAGE_SIZE = [192, 192]
    HEATMAP_SIZE = [48, 48]
    NUM_JOINTS = 21

    # Preprocessing for hands image
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Threshold for keypoint
    kp_thr = np.array([0.15, 0.35, 0.5, 0.6, 0.65,
                       0.35, 0.35, 0.4, 0.5,
                       0.35, 0.35, 0.4, 0.5,
                       0.35, 0.35, 0.4, 0.5,
                       0.5, 0.5, 0.7, 0.7
                       ]).reshape(NUM_JOINTS, 1)

    def __init__(self, yolov5, keypoint_net, device, history_len=4, draw_bbox=False, box_tr=0.5):
        self.yolov5 = yolov5  # OD model (yolov5)
        self.keypoint_net = keypoint_net  # keypoints detector (hrnet)
        self.global_boxes = []  # List for bbox history
        self.history_len = history_len  # Max len of bbox history
        self.pixel_size = None  # Thickness of connection lines and font
        self.draw_bbox = draw_bbox  # Wheater or not draw bboxes
        self.box_tr = box_tr  # Threshold for bbox confidence
        self.device = device  # Device for neural networks
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def detect_keypoints(self, img):

        # Preproccess image for bbox detection
        img, image_for_predict, image_size, ratio_pad = prepare_image(img, self.input_shape)

        # self.image_size = image_size[0]

        # # Set lines and font thickness
        if self.pixel_size == None:
            self.pixel_size = image_size[0] // 300
            self.image_size = image_size[0]
            # self.thickness = self.pixel_size if self.pixel_size > 0 else 1
            # self.font_size = self.thickness
            # self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Get bbox prediction
        lc_info, p_info = yolo_predict(image_for_predict, self.yolov5, self.device,
                                       self.box_tr, self.num_classes)

        p_boxes, p_scores, p_classes = p_info[0], p_info[1], p_info[2]  # Current prediction
        lc_boxes, lc_scores, lc_classes = lc_info[0], lc_info[1], lc_info[2]  # Suspicious prediction

        # Choose bboxes without intersection
        boxes, scores, classes = non_max_suppression_fast(p_boxes, p_scores, p_classes, 0.5)
        lc_boxes, lc_scores, lc_classes = non_max_suppression_fast(lc_boxes, lc_scores, lc_classes, 0.5)

        # Get diffrence between current and suspicious prediction
        if len(lc_boxes) > 0:
            untracked_boxes = choose_untracked_boxes(boxes, lc_boxes)
        else:
            untracked_boxes = []

        # Confirm suspicious prediction by history of bboxes or not
        if len(self.global_boxes) == self.history_len:

            if len(untracked_boxes) > 0:
                boxes, scores, classes = non_max_suppression(boxes, lc_boxes[untracked_boxes], self.global_boxes,
                                                             scores, classes)

            self.global_boxes.append(boxes.copy())
            self.global_boxes = self.global_boxes[1:]
        else:
            self.global_boxes.append(boxes.copy())

        # Rescale prediction to original size
        if len(boxes) > 0:
            boxes[:, :4] = scale_coords(self.input_shape, boxes[:, :4], image_size, ratio_pad).round()

        # Predict and plot keypoints
        img = self.draw_boxes_and_keypoints(img, boxes, scores, classes)

        img = img[:, :, ::-1].copy()

        return img

    def draw_boxes_and_keypoints(self, img, boxes, scores, classes):
        # Plot bbxes and keypoints on original image

        # Loop through all bboxes
        for i, c in reversed(list(enumerate(classes))):

            # predicted_class = name_classes[c]
            predicted_class = 'hand'
            box = boxes[i]
            score = scores[i]

            label = f'{predicted_class} {round(float(score), 2)}'

            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int32') - 1)
            left = max(0, np.floor(left + 0.5).astype('int32') - 1)
            bottom = min(img.shape[0], np.floor(bottom + 0.5).astype('int32') + 1)
            right = min(img.shape[1], np.floor(right + 0.5).astype('int32') + 1)
            box_true = [top, left, bottom, right]

            self.thickness = int(1 / (1.001 - (bottom - top) / self.image_size)) + self.pixel_size
            self.font_size = self.pixel_size
            self.thickness = np.clip(self.thickness, 1, 10)

            # Get keypoints
            preds, pred_mask = self.get_keypoints(img, box_true)
            # Draw keypoints
            img = self.visualize(img, preds, pred_mask)

            if self.draw_bbox:
                text_origin = (left, top + 10)
                cv2.rectangle(img, (left, top), (right, bottom), self.colors[c], self.thickness)
                cv2.putText(img, label, text_origin, self.font, 0.5, self.colors[c], self.font_size, cv2.LINE_AA)

        return img

    def get_keypoints(self, img, box_true):

        '''
        Get keypoints by keypoint_net prediction
        Input:
        - img - original image
        - box_true - list with top, left, bottom, right coordinates
        - transforms - preprocessing operation for keypoint_net input
        - kp_thr - visibility limit for keypoints
        Output:
        - preds - keypoints coordinates
        - pred_mask - keypoints visibility mask
        '''

        # Cut hand image for original image
        top, left, bottom, right = box_true[0], box_true[1], box_true[2], box_true[3]
        hand_img = img[top:bottom, left:right, :].copy()
        # plt.imshow(hand_img)
        # plt.show()

        # Hand image preprocessing
        hand_img = cv2.resize(hand_img, (self.IMAGE_SIZE[0], self.IMAGE_SIZE[1]), interpolation=cv2.INTER_CUBIC)
        scale_x = self.IMAGE_SIZE[1] / (right - left)
        scale_y = self.IMAGE_SIZE[0] / (bottom - top)
        hm_scale = self.IMAGE_SIZE[1] / self.HEATMAP_SIZE[1]
        img_for_predict = self.transforms(hand_img).to(self.device)

        # keypoint_net prediction
        prediction = self.keypoint_net(img_for_predict.unsqueeze(0)).detach().cpu().numpy()

        # Keypoints postprocessing
        preds, pred_mask = self.process_heatmap(prediction)
        preds = preds * hm_scale
        preds[:, 0] = preds[:, 0] / scale_x + left
        preds[:, 1] = preds[:, 1] / scale_y + top

        return preds.astype(np.int32), pred_mask

    def connect_points(self, keypoints_coord, is_visible, hand_img):

        '''
        Function for drawing keypoints connection lines
        Input parameters:
          - keypoints_coord - numpy array of keypoints coordinates ([x, y] * 21)
          - is_visible - flag of visibility of each keypoint ([1(0)] * 21)
          - hand_img - cropped hand from original image
          - thickness - line thickness
        Output parameters:
          - hand_img - hand image connected keypoints by color lines
        '''

        # Loop through all keypoint connection pairs
        for ind, index_pair in enumerate(self.connection_indices):
            # Take coordinates and visibility flag for each keypoint in pair
            point_1 = keypoints_coord[index_pair[0]]
            point_2 = keypoints_coord[index_pair[1]]
            is_vis_1 = is_visible[index_pair[0]]
            is_vis_2 = is_visible[index_pair[1]]

            # Plot line if two points are visible
            if is_vis_1 > 0:
                if is_vis_2 > 0:
                    x1, y1 = int(point_1[0]), int(point_1[1])
                    x2, y2 = int(point_2[0]), int(point_2[1])
                    # color = [x / 255 for x in line_colors[ind]]
                    hand_img = cv2.line(hand_img, (x1, y1), (x2, y2), self.line_colors[ind], self.thickness)

        return hand_img

    def process_heatmap(self, test_target):

        '''
        Interpretation of predicted heatmaps
        Input:
          - heatmaps_pred - output form model ([21, HEATMAP_SIZE, HEATMAP_SIZE])
          - threshold - limit for point visibility
        Output:
          - preds - keypoints in format 21 * [x, y]
          - pred_mask - maximum value on keypoint heatmap (21 * [value])
        '''

        # Get indices and maximim values of each heatmap
        heatmaps_reshaped = test_target.reshape((self.NUM_JOINTS, -1))
        idx = np.argmax(heatmaps_reshaped, 1)
        maxvals = np.amax(heatmaps_reshaped, 1)

        idx = np.expand_dims(idx, -1)
        maxvals = np.expand_dims(maxvals, -1)
        preds = np.tile(idx, (1, 2)).astype(np.float32)

        # Convert indices to coordinates
        preds[:, 0] = (preds[:, 0]) % self.HEATMAP_SIZE[0]
        preds[:, 1] = np.floor((preds[:, 1]) / self.HEATMAP_SIZE[0])

        # Choose only points satisfying the threshold
        pred_mask = np.tile(np.greater(maxvals, self.kp_thr), (1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, pred_mask[:, 0]

    def visualize(self, img, preds, maxvals):

        '''
        Plot keypoints and its connection lines on cropped hand image
        Input:
        - test_img - cropped hand image
        - preds - keypoint prediction (21 * [x, y]) for heatmap with size test_img.size / scale
        - maxvals - maximum value on keypoint heatmap ([value] * 21)
        - thickness - line thickness
        Output:
        - img - image with plotted keypoints and its lines
        '''

        is_visible = maxvals
        is_visible = np.array(is_visible).reshape((self.NUM_JOINTS, 1))

        keypoints_coord = np.array(preds).reshape(-1, 2)

        # Plot point on hand image for each visible keypoint
        for i, point in enumerate(keypoints_coord):
            if is_visible[i] > 0:
                x_p = int(point[0])
                y_p = int(point[1])
                # color = [x / 255 for x in point_colors[i]]
                img = cv2.circle(img, (x_p, y_p), self.thickness + 1, self.point_colors[i], -1)

        # Plot connection lines for keypoints
        img = self.connect_points(keypoints_coord, is_visible, img)

        return img
