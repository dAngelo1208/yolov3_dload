import numpy as np
import torch
from torch import nn

from Darknet import BaseConv, Darknet53


class YOLOBlock(nn.Module):
    def __init__(self, in_chl, out_chl):
        super(YOLOBlock, self).__init__()
        self.conv1 = BaseConv(in_chl, out_chl, ksize=1, stride=1)
        self.conv2 = BaseConv(out_chl, out_chl * 2, ksize=3, stride=1)
        self.conv3 = BaseConv(out_chl * 2, out_chl, ksize=1, stride=1)
        self.conv4 = BaseConv(out_chl, out_chl * 2, ksize=3, stride=1)
        self.conv5 = BaseConv(out_chl * 2, out_chl, ksize=1, stride=1)

    def forward(self, x):
        x1 = self.conv2(self.conv1(x))
        x2 = self.conv4(self.conv3(x1))
        x3 = self.conv5(x2)
        return x3


class YOLOHead(nn.Module):
    def __init__(self, in_chl, mid_chl, out_chl):
        super(YOLOHead, self).__init__()
        self.conv1 = BaseConv(in_chl, mid_chl, ksize=3, stride=1)
        self.conv2 = BaseConv(mid_chl, out_chl, ksize=1, stride=1, with_bn=False, activate=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class YOLOV3(nn.Module):
    def __init__(self, num_class):
        super(YOLOV3, self).__init__()
        self.darknet = Darknet53()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # branch1
        #####
        # 第一个分支：大尺寸
        # 13*13：DarkNet输出1后无需进行尺度的变换，由一次Block和一次Head引入可学习空间即可
        #####
        self.feature1 = YOLOBlock(1024, 512)
        self.out_head1 = YOLOHead(512, 1024, 3 * (num_class + 1 + 4))
        # branch2
        #####
        # 第二个分支：中尺寸
        # 26*26：先将26*26的网络输出2与Block+UpSample的大尺寸特征图cat，然后Block+Head
        #####
        self.cbl1 = self._make_cbl(512, 256, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.feature2 = YOLOBlock(768, 256)
        self.out_head2 = YOLOHead(256, 512, 3 * (num_class + 1 + 4))
        # branch3
        #####
        # 第三个分支：小尺寸
        # 52*52：先将52*52的网络输出3与Block+UpSample的中尺寸特征图cat，然后Block+Head
        #####
        self.cbl2 = self._make_cbl(256, 128, 1)
        self.feature3 = YOLOBlock(384, 128)
        self.out_head3 = YOLOHead(128, 256, 3 * (num_class + 1 + 4))
        self.num_class = num_class

        # anchor set
        self.yolo_anchors = [[[10, 13], [16, 30], [33, 23]],
                             [[30, 61], [62, 45], [59, 119]],
                             [[116, 90], [156, 198], [373, 326]]]
        self.yolo_sample_rate = [8, 16, 32]
        self.sample_rate = np.array(self.yolo_sample_rate)
        self.anchors = torch.from_numpy((np.array(self.yolo_anchors).T / self.sample_rate).T).to(self.device)

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, activate="lrelu")

    def forward(self, x):
        # backbone
        out3, out4, out5 = self.darknet(x)
        # branch1 13X13X255
        feature1 = self.feature1(out5)
        out_large = self.out_head1(feature1)
        # branch2 26X26X255
        cb1 = self.cbl1(feature1)
        up1 = self.upsample(cb1)
        x1_in = torch.cat([up1, out4], 1)
        feature2 = self.feature2(x1_in)
        out_medium = self.out_head2(feature2)
        # branch3 52X52X255
        cb2 = self.cbl2(feature2)
        up2 = self.upsample(cb2)
        x2_in = torch.cat([up2, out3], 1)
        feature3 = self.feature3(x2_in)
        out_small = self.out_head3(feature3)

        return out_large, out_medium, out_small

    def predict(self, out_large, out_medium, out_small):
        """
        param: out_large  [1,255,13,13]
        param: out_medium [1,255,26,26]
        param: out_small  [1,255,52,52]
        """

        # transpose
        #####
        # 通道变换：[n, c, h, w] --> [n, h, w, c] 方便后续的decode解码
        #####
        trans_large = out_large.permute((0, 2, 3, 1))  # [1,13,13,255]
        trans_medium = out_medium.permute((0, 2, 3, 1))  # [1,26,26,255]
        trans_small = out_small.permute((0, 2, 3, 1))  # [1,52,52,255]
        # decode
        pred_small = self.decode(trans_small, i=0)  # [1,52,52,3,85]
        pred_medium = self.decode(trans_medium, i=1)  # [1,26,26,3,85]
        pred_large = self.decode(trans_large, i=2)  # [1,13,13,3,85]

        # out
        #####
        # 将除去num, classes之外的维度全部打平
        #####
        n = pred_small.shape[0]
        c = pred_small.shape[-1]
        out_small = pred_small.view(n, -1, c)
        out_medium = pred_medium.view(n, -1, c)
        out_large = pred_large.view(n, -1, c)

        #####
        # 将三个尺寸的预测结果打平然后concat到一起
        # size(13*13*3+26*26*3+52*52*3) = 10647
        #####
        out_pred = torch.concat([out_small, out_medium, out_large], dim=1)

        return out_pred

    def decode(self, conv_layer, i=0):
        """
        @param1: conv_layer n*h*w*(3*(num_class + 1 + 4)) 13*13/26*26/52*52
        @param2：i：第几个尺寸的特征图
        decode: 对YOLO的输出进行解码，返回以真实图片为坐标的预测框
        """
        n, h, w, c = conv_layer.shape
        conv_output = conv_layer.view(n, h, w, 3, 5 + self.num_class)  # c = 3 * (5 + num_class)
        # divide output
        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position 中心点距离cell左上角的偏移预测量
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset 宽高预测量
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box 置信度
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box 类别预测

        # grid to 13X13 26X26 52X52
        #####
        # mesh grid函数制作坐标 --> grid: [13, 13, 2] --> [1, 13, 13, 1, 2] --> [n, 13, 13, 3(num_anchor), 2]
        # 只为了给中心点偏移量的处理使用
        #####
        yv, xv = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        yv_new = yv.unsqueeze(dim=-1)
        xv_new = xv.unsqueeze(dim=-1)
        xy_grid = torch.concat([xv_new, yv_new], dim=-1)
        # reshape and repeat
        xy_grid = xy_grid.view(1, h, w, 1, 2)  # (13,13,2)-->(1,13,13,1,2)
        xy_grid = xy_grid.repeat(n, 1, 1, 3, 1).float().to(self.device)  # (n,13,13,1,2)--> (n,13,13,3,2)

        # Calculate teh center position and h&w of the prediction box
        #####
        # 根据公式，将网络的预测转化为实际图像坐标
        #####
        # 将sigmoid(中心点偏移量)加到grid坐标上-->中心点坐标（52*52）-->真实中心点坐标（416*416）
        pred_xy = (torch.sigmoid(conv_raw_dxdy) + xy_grid) * self.sample_rate[i]
        # 将exp(宽高预测比值)×锚框-->预测宽高（52*52）-->真实预测宽高（416*416） ##### 这里conv_raw_dwdh[1,52,52,3]中的3，就是3个anchor框
        pred_wh = (torch.exp(conv_raw_dwdh) * self.anchors[i]) * self.sample_rate[i]
        pred_xywh = torch.concat([pred_xy, pred_wh], dim=-1)
        # score and cls
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        return torch.concat([pred_xywh, pred_conf, pred_prob], dim=-1)


if __name__ == "__main__":
    x = torch.randn((1, 3, 416, 416)).cuda()
    model = YOLOV3(3).cuda()
    cnt = 0
    out = model(x)
    out_large, out_medium, out_small = model(x)
    print(out_large.shape)
    print(out_medium.shape)
    print(out_small.shape)
    pred_box = model.predict(out_large, out_medium, out_small)
    # res = postprocess_boxes
