import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from config import config
from backbone import resnet50
from backbone import fpn
from det_opr.anchors_generator import generate_anchors_opr
from det_opr.find_top_rpn_proposals import find_top_rpn_proposals
from det_opr.fpn_anchor_target import fpn_anchor_target, fpn_rpn_reshape
from det_opr.fpn_roi_target import fpn_roi_target
from det_opr.loss_opr import smooth_l1_loss
from det_opr.bbox_opr import bbox_transform_inv_opr, bbox_transform_opr
from layers.pooler import ROIPooler
from det_tools.img_utils import pad_tensor_to_multiple_number
from det_tools.load_utils import _init_backbone

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = resnet50.ResNet50(1)
        _init_backbone(self.resnet50, config.init_weights)
        self.FPN = fpn.FPN(self.resnet50)
        self.RPN = RPN()
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):
        if self.training:
            image = image[:, :, :int(im_info[0, 0]), :int(im_info[0, 1])]
            image = image - torch.tensor(config.image_mean[None, :, None, None],
                    dtype=image.dtype, device=image.device)
            image = pad_tensor_to_multiple_number(image, 64)
            return self._forward_train(image, im_info, gt_boxes)
        else:
            image = image[:, :, :int(im_info[0, 0]), :int(im_info[0, 1])]
            image = image - torch.tensor(config.image_mean[None, :, None, None],
                    dtype=image.dtype, device=image.device)
            image = pad_tensor_to_multiple_number(image, 64)
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        ## stride: 64,32,16,8,4
        rpn_rois, rpn_rois_inds, loss_dict_rpn = \
            self.RPN(fpn_fms, im_info, gt_boxes)
        with torch.no_grad():
            rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                rpn_rois, rpn_rois_inds, im_info, gt_boxes)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
            rcnn_labels, rcnn_bbox_targets)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois, rpn_rois_inds = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois)
        pred_bbox[:, :-1] /= im_info[0, 2]
        return pred_bbox.cpu().detach()

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        rpn_channel = 256
        num_cell_anchors = config.num_cell_anchors
        self.rpn_conv = nn.Conv2d(256, rpn_channel, kernel_size=3, stride=1, padding=1)
        self.rpn_cls_score = nn.Conv2d(rpn_channel, num_cell_anchors, kernel_size=1, stride=1)
        self.rpn_bbox_offsets = nn.Conv2d(rpn_channel, num_cell_anchors * 4, kernel_size=1, stride=1)

        for l in [self.rpn_conv, self.rpn_cls_score, self.rpn_bbox_offsets]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        self.anchor_scales = config.anchor_scales[::-1]

    def forward(self, features, im_info, boxes=None):
        assert len(features) == config.num_anchor_scales

        # get the predict results
        pred_cls_score_list = []
        pred_bbox_offsets_list = []
        all_anchors_list = []
        for x in features:
            t = F.relu(self.rpn_conv(x))
            pred_cls_score_list.append(self.rpn_cls_score(t))
            pred_bbox_offsets_list.append(self.rpn_bbox_offsets(t))
        # get anchors
        max_fm_stride = 2 ** (len(features) + 1)
        fm_stride = max_fm_stride
        anchor_scales = config.anchor_scales[::-1]
        for i in range(config.num_anchor_scales):
            anchor_scale = [anchor_scales[i]]
            layer_anchors = generate_anchors_opr(features[i], fm_stride, anchor_scale,
                        config.anchor_aspect_ratios, config.anchor_base_size)
            fm_stride = fm_stride // 2
            all_anchors_list.append(layer_anchors)
        # sample from the predictions
        with torch.no_grad():
            rpn_rois, rpn_rois_inds = find_top_rpn_proposals(
                self.training, pred_bbox_offsets_list, pred_cls_score_list,
                all_anchors_list, im_info)

        if self.training:
            with torch.no_grad():
                rpn_labels, rpn_bbox_targets = fpn_anchor_target(
                    boxes, im_info, all_anchors_list)
            pred_cls_score, pred_bbox_offsets = fpn_rpn_reshape(
                pred_cls_score_list, pred_bbox_offsets_list)

            # rpn loss
            valid_masks = rpn_labels >= 0
            objectness_loss = F.binary_cross_entropy_with_logits(
                pred_cls_score[valid_masks],
                rpn_labels[valid_masks].to(torch.float32),
                reduction="sum")
            pos_masks = rpn_labels == 1
            localization_loss = smooth_l1_loss(
                pred_bbox_offsets[pos_masks],
                rpn_bbox_targets[pos_masks],
                config.rpn_smooth_l1_beta, reduction="sum")
            normalizer = 1.0 / (config.train_batch_per_gpu * config.num_sample_anchors)
            loss_rpn_cls = objectness_loss * normalizer  # cls: classification loss
            loss_rpn_loc = localization_loss * normalizer
            loss_dict = {}
            loss_dict['loss_rpn_cls'] = loss_rpn_cls
            loss_dict['loss_rpn_loc'] = loss_rpn_loc
            return rpn_rois, rpn_rois_inds, loss_dict
        else:
            return rpn_rois, rpn_rois_inds

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # pooler
        stride = [4, 8, 16, 32]
        pooler_scales = tuple(1.0 / k for k in stride)
        self.box_pooler = ROIPooler(
            output_size=7,
            scales=pooler_scales,
            sampling_ratio=0,
            pooler_type='ROIAlignV2',
        )
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, (config.num_classes - 1) * 4)
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None):
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        pool_features = self.box_pooler(fpn_fms, rcnn_rois)
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        roi_feature = F.relu(self.fc1(flatten_feature))
        roi_feature = F.relu(self.fc2(roi_feature))
        pred_cls = self.pred_cls(roi_feature)
        pred_delta = self.pred_delta(roi_feature)
        if self.training:
            # loss for regression
            labels = labels.flatten()
            fg_inds = torch.nonzero(labels>0).flatten()
            loss_rcnn_reg = smooth_l1_loss(
                pred_delta[fg_inds],
                bbox_targets[fg_inds],
                config.rcnn_smooth_l1_beta, 'sum')
            loss_rcnn_reg = loss_rcnn_reg / pred_delta.shape[0]
            # loss for classification
            loss_rcnn_cls = F.cross_entropy(pred_cls, labels.long(), reduction="mean")
            loss_dict = {}
            loss_dict['loss_rcnn_reg'] = loss_rcnn_reg
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            return loss_dict
        else:
            pred_scores = F.softmax(pred_cls, dim=-1)
            pred_bbox = restore_bbox(rcnn_rois[:, 1:5], pred_delta, True)
            pred_bbox = torch.cat([pred_bbox, pred_scores[:, 1].reshape(-1,1)], dim=1)
            return pred_bbox

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]
            ).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]
            ).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

def _init_weights(layer):
    nn.init.kaiming_uniform_(layer.weight, a=1)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    net = Network()
    data = torch.Tensor(np.random.rand(2, 3, 800, 800))
    box = torch.Tensor(np.random.rand(2, 8, 5)*100)
    im_info = torch.tensor([[800,800,0,0,0,8],[800,800,0,0,0,8]]).cuda()
    net.cuda()
    data = data.cuda()
    box = box.cuda()
    net.train()
    output = net(data, im_info, box)
    print(output)
