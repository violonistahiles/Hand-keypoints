import numpy as np
import cv2

import torch
import torch.nn as nn


def prepare_image(img, input_shape):
    # Preprocessing image for yolo

    img = img[:, :, ::-1].copy()

    # Reshape image
    h, w = input_shape
    ih, iw = img.shape[:2]
    image_size = np.array([ih, iw])
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_for_predict = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # Paste reshaped image in gray image with size input_shape
    new_image = np.full(shape=(h, w, 3), fill_value=128)
    h_start = (h - nh) // 2
    w_start = (w - nw) // 2
    new_image[h_start:h_start + nh, w_start:w_start + nw] = image_for_predict
    image_for_predict = np.expand_dims(new_image.transpose(2, 0, 1), 0) / 255.0

    ratio_pad = [[scale], [w_start, h_start]]

    return img, image_for_predict, image_size, ratio_pad


def yolo_predict(image_for_predict, yolov5, device, box_tr, num_classes):
    # Get bbox prediction
    p_boxes = []
    p_scores = []
    p_classes = []
    # print(len(image_for_predict))

    img_torch = torch.from_numpy(image_for_predict).to(device).float()
    # img_torch = img_torch.permute(0,3,1,2)

    # Get bbox prediction
    prediction = yolov5(img_torch)[0]

    pred_numpy = prediction.detach().cpu().numpy().squeeze()
    # print(pred_numpy.shape)
    # Postprocces bboxes
    box_conf = pred_numpy[:, 4:5]
    box_class = pred_numpy[:, 5:]
    boxes_score = np.reshape(box_conf * box_class, (-1, num_classes))

    # Real prediction with adjusted confidence
    mask_x, mask_y = np.where(boxes_score >= box_tr)
    p_boxes = np.zeros((len(mask_x), 4))
    p_boxes[:, 0:2] = pred_numpy[mask_x, 0:2] - pred_numpy[mask_x, 2:4] // 2
    p_boxes[:, 2:4] = pred_numpy[mask_x, 0:2] + pred_numpy[mask_x, 2:4] // 2
    p_scores = boxes_score[mask_x, mask_y]
    p_classes = np.zeros_like(p_scores, 'int32') + mask_y

    # Low limit confidece prediction for missboxes correction
    mask_x, mask_y = np.where(boxes_score >= 0.05)
    sus_boxes = np.zeros((len(mask_x), 4))
    sus_boxes[:, 0:2] = pred_numpy[mask_x, 0:2] - pred_numpy[mask_x, 2:4] // 2
    sus_boxes[:, 2:4] = pred_numpy[mask_x, 0:2] + pred_numpy[mask_x, 2:4] // 2
    sus_scores = boxes_score[mask_x, mask_y]
    sus_classes = np.zeros_like(sus_scores, 'int32') + mask_y

    sus_info = [sus_boxes, sus_scores, sus_classes]
    p_info = [p_boxes, p_scores, p_classes]

    return sus_info, p_info
