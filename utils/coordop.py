import numpy as np


def non_max_suppression_fast(boxes, scores, classes, overlapThresh):
    # Select only one bbox among all bboxes with intesection above the threshold
    if len(boxes) == 0:  # If there are no bboxes
        return [], [], []

    pick = []  # Choosen bboxes indices

    # Top left and right bottom coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Sort bboxes by score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # Choose only one bbox with intersection heighter then overlapThresh
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick], scores[pick], classes[pick]


def choose_untracked_boxes(boxes, lc_bboxes):
    '''
    Function for identifying untracked bboxes
    Input:
      - boxes - bboxes predicted by yolo
      - lc_bboxes - bboxes getted from by yolo with low limit confidece (lc_bboxes)
    Output:
       - untracked_boxes - indices of suspicious predictions, whitch might be good
    '''

    # Top left and right bottom coordinates of lc_bboxes
    xt1 = lc_bboxes[:, 0]
    yt1 = lc_bboxes[:, 1]
    xt2 = lc_bboxes[:, 2]
    yt2 = lc_bboxes[:, 3]
    # Indices of trackers bboxes whitch is not intersect with yolo bboxes
    untracked_boxes = set(range(len(lc_bboxes)))

    for i in range(len(boxes)):
        # Top left and right bottom coordinates of current yolo bbox
        x1 = boxes[i, 0]
        y1 = boxes[i, 1]
        x2 = boxes[i, 2]
        y2 = boxes[i, 3]

        # Find intersection of current yolo bbox with lc_bboxes
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        xx1 = np.maximum(x1, xt1)
        yy1 = np.maximum(y1, yt1)
        xx2 = np.minimum(x2, xt2)
        yy2 = np.minimum(y2, yt2)

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area
        idxs = np.where(overlap > 0.5)[0]

        # If there is intersection, remove lc_bbox from tracking list
        if len(idxs) != 0:
            untracked_boxes -= set(idxs)

    untracked_boxes = list(untracked_boxes)

    return untracked_boxes


def non_max_suppression(boxes, lc_bboxes, global_boxes, scores, classes):
    '''
    Function for extention boxes by good lc_bboxes
    Input:
      - boxes - bboxes predicted by yolo
      - lc_bboxes - bboxes getted from by yolo with low limit confidece (lc_bboxes)
      - global_boxes - history of yolo prediction n steps back
      - scores - yolo bboxes scores
      - classes - yolo bboxes classes
    Output:
       - boxes - yolo bboxes supplemented by lc_bboxes
       - scores - yolo scores supplemented by lc_bboxes
       - classes - yolo classes supplemented by lc_bboxes
    '''

    # Indices of lc_bboxes whitch have no intersection with yolo bboxes
    untracked_boxes = set(range(len(lc_bboxes)))
    gl_boxes = []
    for i in range(len(global_boxes)):
        gl_boxes.extend(global_boxes[i])

    gl_boxes = np.array(gl_boxes)
    # print(gl_boxes.shape, untracked_boxes, len(untracked_boxes))
    if len(gl_boxes) == 0:
        return boxes, scores, classes

    # Top left and right bottom coordinates of global bboxes
    xt1 = gl_boxes[:, 0]
    yt1 = gl_boxes[:, 1]
    xt2 = gl_boxes[:, 2]
    yt2 = gl_boxes[:, 3]

    for i in range(len(lc_bboxes)):
        # Top left and right bottom coordinates of current lc_bbox
        x1 = lc_bboxes[i, 0]
        y1 = lc_bboxes[i, 1]
        x2 = lc_bboxes[i, 2]
        y2 = lc_bboxes[i, 3]

        # Find intersection of current lc_bbox with bboxes from history
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        xx1 = np.maximum(x1, xt1)
        yy1 = np.maximum(y1, yt1)
        xx2 = np.minimum(x2, xt2)
        yy2 = np.minimum(y2, yt2)

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area
        idxs = np.where(overlap > 0.25)[0]

        # If there are no intersections, remove lc_bbox from tracking list
        if len(idxs) == 0:
            untracked_boxes -= set([i])

    # If there are lc_bboxes without yolo intersection, add them in bbox predictions
    untracked_boxes = list(untracked_boxes)

    if len(untracked_boxes) > 0:
        if len(boxes) > 0:
            for i in range(len(untracked_boxes)):
                box_to_add = np.array([lc_bboxes[untracked_boxes[i]]])
                boxes = np.concatenate([boxes, box_to_add], axis=0)
                scores = np.concatenate([scores, [1]], axis=0)
                classes = np.concatenate([classes, [0]], axis=0)
        else:
            boxes = np.array(lc_bboxes)
            boxes = boxes[untracked_boxes]
            scores = np.ones((len(boxes),))
            classes = np.zeros_like(scores).astype(np.int32)

    return boxes, scores, classes


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
