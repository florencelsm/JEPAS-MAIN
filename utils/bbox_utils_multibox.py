import torch
import cv2
from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    return iou - (area - union) / area

def plot_bbox(images, output, gt_bboxes, save_path, global_step, train=False):
    H, W = images.shape[-2:]
    images = images[0].permute(1, 2, 0).cpu().numpy()
    images = cv2.cvtColor((images), cv2.COLOR_RGB2BGR) * 255
    bboxes = output["pred_boxes"].cpu().numpy()
    scores = output["pred_scores"].cpu().numpy()

    for i, (bbox,score) in enumerate(zip(bboxes, scores)):
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(images, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(images, f"{score:.2f}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    for i, bbox in enumerate(gt_bboxes):
        for bbox_i in bbox:
            x0, y0, x1, y1 = bbox_i
            x0, y0, x1, y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
            cv2.rectangle(images, (x0, y0), (x1, y1), (0, 0, 255), 2)
        if train:
            break
    save_name = 'train' if train else 'val'
    cv2.imwrite(f"{save_path}/output_image_{save_name}_{global_step}.png", images)

@torch.no_grad()
def compute_metrics(pred_output, gt_boxes_list, iou_threshold=0.5, image_shape=None):
    """
    Computes detection metrics for one image.

    Args:
        pred_output: dict with
            - pred_boxes: (num_preds, 4) xyxy in absolute coords
            - pred_scores: (num_preds,) confidence scores
        gt_boxes_list: list of (num_gt, 4) tensors [xyxy] in normalized coords
        iou_threshold: IoU threshold for a TP
        image_shape: (H, W) of the image, required if GT boxes are normalized
    """
    pred_boxes = pred_output['pred_boxes']
    pred_scores = pred_output['pred_scores']

    # flatten GT list
    if len(gt_boxes_list) > 0:
        gt_boxes = gt_boxes_list[0] if len(gt_boxes_list) == 1 else torch.cat(gt_boxes_list, dim=0)
    else:
        gt_boxes = torch.empty(0, 4, device=pred_boxes.device)

    # scale GT boxes from normalized to absolute if needed
    if gt_boxes.numel() > 0 and image_shape is not None:
        img_h, img_w = image_shape
        scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=gt_boxes.device)
        gt_boxes = gt_boxes * scale_factor

    # edge cases
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mean_iou': 0.0,
            'num_predictions': len(pred_boxes),
            'num_gt': len(gt_boxes)
        }

    # IoU matrix (num_preds, num_gt)
    ious, _ = box_iou(pred_boxes, gt_boxes)

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
    matched_ious = []

    # sort predictions by confidence
    sorted_indices = torch.argsort(pred_scores, descending=True)

    tp = 0
    for i in sorted_indices:
        pred_idx = i.item()
        pred_ious = ious[pred_idx]  # IoU with all GT boxes

        best_iou, best_gt_idx = torch.max(pred_ious, dim=0)
        best_gt_idx = best_gt_idx.item()

        if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
            tp += 1
            matched_gt[best_gt_idx] = True
            matched_ious.append(best_iou.item())

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou,
        'num_predictions': len(pred_boxes),
        'num_gt': len(gt_boxes)
    }