import colorsys
import random

import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    try:
        pred_bbox = pred_bbox.detach().numpy()  # (1,10647,85)
    except TypeError:
        pred_bbox = pred_bbox.cpu().detach().numpy()
    n, box_num, c = pred_bbox.shape

    cur_pred = pred_bbox[0, :, :]
    pred_xywh = cur_pred[:, 0:4]
    pred_conf = cur_pred[:, 4]
    pred_prob = cur_pred[:, 5:]
    # 1.(x,y,w,h) --> (x1,y1,x2,y2)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2.(x1,y1,x2,y2) -> (x1_org, y1_org, x2_org, y2_org)(org*org)
    #####
    # 与image_process中的处理过程保持一致：[计算最小缩放率进行Resize --> 对缩放后非正方形的区域进行128值（中值）Padding]反向
    #####
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size * 1.0 / org_w, input_size * 1.0 / org_h)
    dw = (input_size - resize_ratio * org_w) / 2.0
    dh = (input_size - resize_ratio * org_h) / 2.0

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    #####
    # 将原图大小以外的框掩盖掉
    #####
    # 对中心点坐标，只取＞0的部分；对宽高坐标，要求小于原图的宽高
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # 对中心点在宽高之外的部分通过掩码的方式排除；
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)  # 找到每一个框预测的类别（索引）
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]  # 将每一个预测框的分类prob根据上面的索引取出来，然后×置信度
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def image_preprocess(image, target_size, gt_boxes=None):
    """
    对原图进行Resize和padding：计算最小缩放率进行Resize --> 对缩放后非正方形的区域进行128值（中值）Padding
    """
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    #####
    # 除2：对图片进行填补，所以左右两边进行平均的填补，填补完进行归一化
    #####
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    # image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        #####
        # 如果是训练图片，将ground_truth框也进行缩放，后面计算loss使用
        #####
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def img_loader(photo_file, input_w, input_h):
    img = image_preprocess(cv2.imread(photo_file), [416, 416])
    img = Image.fromarray(np.uint8(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB)))

    # img = Image.open(photo_file)
    img_w, img_h = img.size
    img = img.resize((input_w, input_h))
    # 返回指定大小的图片张量和图片原始的宽高

    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    img = trans(img)

    return img, img_w, img_h


def bboxes_iou(boxes1, boxes2):
    """
    :func: used in NMS below
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)

    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, which class)（原图尺寸）

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))  # 预测框的所有预测类别取出并set
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]  # 将预测为此类的预测框取出
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        # 对所有预测框进行[最佳框-->iou极大抑制]的循环，直到最终没有框为止
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to score order A
            max_ind = np.argmax(cls_bboxes[:, 4])  # 这些框中score(conf*prob)最高的框的索引
            best_bbox = cls_bboxes[max_ind]  # 拿到score最高的框
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])  # 从boxes中去掉刚刚处理的得分最高的框，方便遍历
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0  # 对于org-nms，只需要将与最佳框重合度高的其他框掩掉即可

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))  # 对于soft-nms，将与最佳框重合度高的其他框赋予权重，继续进行迭代

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.  # 将score非0的框保留
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def read_class_names(class_file_name="./class_names/coco_name.txt"):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, CLASSES="./class_names/coco_name.txt", show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    # print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""

            if tracking: score_str = " " + str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                          thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image
