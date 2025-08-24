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

    for i, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(images, (x0, y0), (x1, y1), (0, 255, 0), 2)
        if train:
            break
    
    for i, bbox in enumerate(gt_bboxes):
        x0, y0, x1, y1 = bbox
        x0, y0, x1, y1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        cv2.rectangle(images, (x0, y0), (x1, y1), (0, 0, 255), 2)
        if train:
            break
    save_name = 'train' if train else 'val'
    cv2.imwrite(f"{save_path}/output_image_{save_name}_{global_step}.png", images)

@torch.no_grad()
def compute_metrics(pred_output, gt_boxes, iou_threshold=0.5, image_shape=None):
    pred_boxes = pred_output['pred_boxes']
    if image_shape is not None:
        img_h, img_w = image_shape
        scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=bboxes.device)
        gt_boxes = gt_boxes * scale_factor
    
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mean_iou': 0.0,
            'num_predictions': len(pred_boxes),
            'num_gt': len(gt_boxes)
        }

    ious, _ = box_iou(pred_boxes, gt_boxes)
    iou = ious.item()
    tp = 1 if iou >= iou_threshold else 0
    
    precision = tp / 1.0
    recall = tp / float(len(gt_boxes))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': iou,
        'num_predictions': len(pred_boxes),
        'num_gt': len(gt_boxes)
    }