import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class MatchingLoss(nn.Module):
    def __init__(self, cost_bbox: float = 5.0, cost_giou: float = 2.0, **kwargs):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.weight_dict = {'loss_bbox': cost_bbox, 'loss_giou': cost_giou}
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def _loss_boxes(self, outputs, targets, num_boxes):
        targets = box_xyxy_to_cxcywh(targets)
        loss_bbox = F.l1_loss(outputs, targets, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(outputs),
                                                       box_cxcywh_to_xyxy(targets)))
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou
    
    def forward(self, outputs, targets):
        losses = {}
        num_boxes = max(len(outputs['pred_boxes']), 1)
        box_losses, giou_losses = self._loss_boxes(outputs['pred_boxes'], targets, num_boxes)
        losses['loss_bbox'] = box_losses
        losses['loss_giou'] = giou_losses
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        return total_loss