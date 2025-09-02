import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class MatchingLoss(nn.Module):
    def __init__(self, cost_bbox: float = 5.0, cost_giou: float = 2.0, cost_cls: float = 1.0, num_classes: int = 2, eos_coef: float = 0.1):
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_cls = cost_cls
        self.weight_dict = { "loss_bbox": cost_bbox,
                             "loss_giou": cost_giou,
                             "loss_ce": cost_cls,}
        empty_weight = torch.ones(num_classes).cuda()
        empty_weight[0] = eos_coef  # background = 0
        self.register_buffer("empty_weight", empty_weight)

    def _hungarian_match(self, pred_boxes, pred_logits, tgt_boxes):
        num_gt = tgt_boxes.shape[0]
        
        if num_gt == 0:
            return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        
        out_prob = pred_logits.softmax(-1)
        
        cost_cls = -out_prob[:, 1:2]
        cost_cls = cost_cls.expand(-1, num_gt)
        cost_bbox = torch.cdist(pred_boxes, tgt_boxes, p=1)

        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou_matrix = generalized_box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
        cost_giou = 1 - giou_matrix
        
        C = (self.cost_bbox * cost_bbox +
             self.cost_cls * cost_cls +
             self.cost_giou * cost_giou)
        
        C = C.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C)
        
        return (torch.as_tensor(row_ind, dtype=torch.long, device=pred_boxes.device),
                torch.as_tensor(col_ind, dtype=torch.long, device=pred_boxes.device))

    def _loss_boxes(self, pred_boxes, tgt_boxes, indices):
        if len(indices[0]) == 0:
            device = pred_boxes.device
            return torch.tensor(0., device=device, requires_grad=True), torch.tensor(0., device=device, requires_grad=True)
        matched_pred = pred_boxes[indices[0]]
        matched_tgt = tgt_boxes[indices[1]]
        loss_bbox = F.l1_loss(matched_pred, matched_tgt, reduction="mean")
        pred_xyxy = box_cxcywh_to_xyxy(matched_pred)
        tgt_xyxy = box_cxcywh_to_xyxy(matched_tgt)
        giou_matrix = generalized_box_iou(pred_xyxy, tgt_xyxy)
        loss_giou = 1 - torch.diag(giou_matrix).mean()
        return loss_bbox, loss_giou

    def _loss_labels(self, pred_logits, indices):
        device = pred_logits.device
        target_classes = torch.zeros(pred_logits.shape[0], dtype=torch.long, device=device)
        if len(indices[0]) > 0:
            target_classes[indices[0]] = 1
        loss_ce = F.cross_entropy(pred_logits, target_classes, weight=self.empty_weight)
        return loss_ce

    def forward(self, outputs, targets):
        batch_size = outputs["pred_boxes"].shape[0]
        device = outputs["pred_boxes"].device
        
        total_losses = {"loss_bbox": torch.tensor(0., device=device),
                       "loss_giou": torch.tensor(0., device=device),
                       "loss_ce": torch.tensor(0., device=device)}
        
        valid_samples = 0
        total_matched_boxes = 0
        
        for i in range(batch_size):
            pred_boxes = outputs["pred_boxes"][i]
            pred_logits = outputs["pred_logits"][i]
            tgt_boxes = targets[i]
            tgt_boxes_cxcywh = box_xyxy_to_cxcywh(tgt_boxes)
            indices = self._hungarian_match(pred_boxes, pred_logits, tgt_boxes_cxcywh)
            loss_bbox, loss_giou = self._loss_boxes(pred_boxes, tgt_boxes_cxcywh, indices)
            loss_ce = self._loss_labels(pred_logits, indices)
            total_losses["loss_bbox"] += loss_bbox
            total_losses["loss_giou"] += loss_giou
            total_losses["loss_ce"] += loss_ce
            valid_samples += 1
            total_matched_boxes += len(indices[0])
        
        for k in total_losses:
            if valid_samples > 0:
                total_losses[k] = total_losses[k] / valid_samples
        
        total_loss = sum(total_losses[k] * self.weight_dict[k] for k in total_losses)

        return total_loss