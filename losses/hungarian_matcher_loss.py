import torch
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class MatchingLoss(nn.Module):
    def __init__(self, cost_logits: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0, **kwargs):
        super().__init__()
        self.cost_logits = cost_logits
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.weight_dict = {'loss_bbox': cost_bbox, 'loss_giou': cost_giou, 'loss_logit': cost_logits}
        assert cost_logits != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    
    def _convert_gt_for_matcher(self, gt_boxes_list):
        targets = []
        
        for sample_boxes in gt_boxes_list:
            if isinstance(sample_boxes, list):
                if sample_boxes:
                    sample_boxes_tensor = torch.cat(sample_boxes, dim=0)  # [N, 4]
                else:
                    sample_boxes_tensor = torch.empty(0, 4)
            else:
                sample_boxes_tensor = sample_boxes
            
            if len(sample_boxes_tensor) > 0:
                sample_boxes_cxcywh = box_xyxy_to_cxcywh(sample_boxes_tensor)
                num_objects = len(sample_boxes_cxcywh)
                logits_labels = torch.ones(num_objects, device=sample_boxes_tensor.device)
                targets.append({'boxes': sample_boxes_cxcywh, 'logits': logits_labels})
            else:
                device = sample_boxes_tensor.device if hasattr(sample_boxes_tensor, 'device') else torch.device('cpu')
                targets.append({'boxes': torch.empty(0, 4, device=device),
                                'logits': torch.empty(0, device=device)})
        return targets
    
    @torch.no_grad()
    def _hungarian_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        out_logits = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_logits = torch.cat([v["logits"] for v in targets])
        
        if len(tgt_bbox) == 0:
            return [(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)) 
                   for _ in range(bs)]
        
        cost_logits = 1 - out_logits.squeeze(-1)
        cost_logits = cost_logits.unsqueeze(-1).expand(-1, len(tgt_logits))
        
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))
        
        C = self.cost_bbox * cost_bbox + self.cost_logits * cost_logits + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        if len(src_boxes) == 0:
            return torch.tensor(0.0, device=outputs['pred_boxes'].device, requires_grad=True)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                       box_cxcywh_to_xyxy(target_boxes)))
        loss_giou = loss_giou.sum() / num_boxes
        return loss_bbox, loss_giou
    
    def _loss_logits(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        bs, num_queries = outputs['pred_logits'].shape[:2]
        target_logits = torch.zeros((bs, num_queries), 
                                      device=outputs['pred_logits'].device, 
                                      dtype=torch.float32)
        idx = self._get_src_permutation_idx(indices)
        if len(idx[0]) > 0:
            target_logits[idx] = 1.0
        pred_logits = outputs['pred_logits'].squeeze(-1)
        loss_obj = F.binary_cross_entropy_with_logits(pred_logits, target_logits, reduction='none')
        return loss_obj.mean()
    
    def forward(self, outputs, targets):
        losses = {}
        gt_boxes_tensor = self._convert_gt_for_matcher(targets)
        indices = self._hungarian_forward(outputs, gt_boxes_tensor)
        num_boxes = max(sum(len(t['boxes']) for t in gt_boxes_tensor), 1)
        box_losses, giou_losses = self._loss_boxes(outputs, gt_boxes_tensor, indices, num_boxes)
        logit_losses = self._loss_logits(outputs, gt_boxes_tensor, indices, num_boxes)
        losses['loss_bbox'] = box_losses
        losses['loss_giou'] = giou_losses
        losses['loss_logit'] = logit_losses
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys() if k in self.weight_dict)
        return total_loss