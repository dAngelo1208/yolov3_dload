import numpy as np
import cv2
import torch
from yolov3 import YOLOV3
from utils import image_preprocess, img_loader, postprocess_boxes, nms, draw_bbox


def read_param_from_file(yolo_ckpt, model):
    wf = open(yolo_ckpt, 'rb')
    major, minor, vision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    print("version major={} minor={} vision={} and pic_seen={}".format(major, minor, vision, seen))

    model_dict = model.state_dict()
    key_list = [key for key in model_dict.keys()]
    num = 6
    length = int(len(key_list) // num)
    pre_index = 0
    for i in range(length + 2):
        cur_list = key_list[pre_index:pre_index + num]
        conv_name = cur_list[0]
        conv_layer = model_dict[conv_name]
        filters = conv_layer.shape[0]
        in_dim = conv_layer.shape[1]
        k_size = conv_layer.shape[2]
        conv_shape = (filters, in_dim, k_size, k_size)
        # print("i={} and list={} amd conv_name={} and shape={}".format(i, cur_list,conv_name,conv_shape))
        if len(cur_list) == 6:  # with bn
            # darknet bn param:[bias,weight,mean,variance]
            bn_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[2]].data.copy_(torch.from_numpy(bn_bias))
            bn_weight = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[1]].data.copy_(torch.from_numpy(bn_weight))
            bn_mean = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[3]].data.copy_(torch.from_numpy(bn_mean))
            bn_variance = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[4]].data.copy_(torch.from_numpy(bn_variance))
            # darknet conv param:(out_dim, in_dim, height, width)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape)
            model_dict[cur_list[0]].data.copy_(torch.from_numpy(conv_weights))
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            model_dict[cur_list[1]].data.copy_(torch.from_numpy(conv_bias))
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape)
            model_dict[cur_list[0]].data.copy_(torch.from_numpy(conv_weights))

        pre_index += num
        if i in [57, 65, 73]:
            num = 2
        else:
            num = 6
    assert len(wf.read()) == 0, 'failed to read all data'


def detect(image_path):
    input_size = 416
    num_class = 3
    iou_threshold = 0.45  # nms中用于抑制与最佳框(score最高)重合度高的预测框 防止一obj多框
    score_threshold = 0.3  # 在对预测框处理完成后
    # rectangle_colors = (255, 0, 0)

    yolo_ckpt = './muskDetectWeight/best.pt'
    model = YOLOV3(num_class).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.eval()
    # read_param_from_file(yolo_ckpt, model)
    read_param_from_master(yolo_ckpt, model)

    # image_path = './IMAGES/kite.jpg'
    # image_path = './IMAGES/ocean.jpg'

    original_image = cv2.imread(image_path)
    image_data = cv2.cvtColor(img_loader(image_path, 416, 416)[0].detach().numpy().transpose((1, 2, 0)),
                              cv2.COLOR_RGB2BGR)

    # print(original_image.shape)
    # image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    cv2.imshow("image processed", image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_data = image_data.transpose((2, 0, 1))
    image_data = image_data[np.newaxis, ...].astype(np.float32)  # (new_axis, 3, 416, 416)
    input_tensor = torch.from_numpy(image_data).float().to(
        torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    out_l, out_m, out_s = model(input_tensor)

    # decode
    out_pred = model.predict(out_l, out_m, out_s)
    # post process
    bboxes = postprocess_boxes(out_pred, original_image, input_size=input_size, score_threshold=score_threshold)
    print("before nms box num is ", len(bboxes))
    bboxes = nms(bboxes, iou_threshold, method='nms')
    print("after nms box num is ", len(bboxes))

    # draw
    # image = draw_bbox(original_image, bboxes, CLASSES='./class_names/coco_name.txt', rectangle_colors=rectangle_colors)
    # image = draw_bbox(original_image, bboxes, CLASSES='./class_names/coco_name.txt')
    image = draw_bbox(original_image, bboxes, CLASSES='./class_names/musk_name.txt')
    cv2.imshow("draw", image)
    cv2.imwrite("result/draw.jpg", image)
    cv2.waitKey(0)


def read_param_from_master(pt_path, model):
    ckp = torch.load(pt_path, map_location=torch.device('cpu'))
    model_dict_pt = ckp['model']
    model_dict = model.state_dict()

    ks_pt, vs_pt = list(model_dict_pt.keys()), list(model_dict_pt.values())
    ks, vs = list(model_dict.keys()), list(model_dict.values())
    idx_jump = 0
    dict_update = {}
    for k, v in zip(ks, vs):
        if '_head' in k and '.conv2' in k:
            idx_jump += 1
            # 更新Head层参数
            # dict_update.update({k: vs_pt[-idx_jump - 1]})
            continue

        current_idx = ks.index(k)
        k_pt, v_pt = ks_pt[int(current_idx - idx_jump)], vs_pt[int(current_idx - idx_jump)]
        dict_update.update({k: v_pt})

    # 更新Yolo Head层参数
    dict_update['out_head1.conv2.conv.weight'] = vs_pt[-2]
    dict_update['out_head1.conv2.conv.bias'] = vs_pt[-1]
    dict_update['out_head2.conv2.conv.weight'] = vs_pt[-4]
    dict_update['out_head2.conv2.conv.bias'] = vs_pt[-3]
    dict_update['out_head3.conv2.conv.weight'] = vs_pt[-6]
    dict_update['out_head3.conv2.conv.bias'] = vs_pt[-5]

    model_dict.update(dict_update)
    model.load_state_dict(model_dict)
    print("pre-trained parameters loaded.")


if __name__ == "__main__":
    detect('./IMAGES/ocean.jpg')

    # model = YOLOV3(3).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    # print(model)
    # read_param_from_master("./muskDetectWeight/best.pt", model)
