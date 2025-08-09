import librosa

def load_audio(waveform_path, sample_rate):
    waveform, sample_rate = librosa.load(waveform_path, sr=sample_rate)
    return waveform, sample_rate

def unnormalize_bbox(bbox, original_size):
    W, H = original_size
    x_min, y_min, x_max, y_max = bbox
    return [int(x_min * W),
            int(y_min * H),
            int(x_max * W),
            int(y_max * H)]

def scale_bbox(bbox, original_size, image_shape):
    New_H, New_W = image_shape
    W, H = original_size
    scale_x = New_W / W
    scale_y = New_H / H
    return [int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),]

def get_bbox_ratio_img(bbox, image_shape):
    H, W = image_shape
    x_min, y_min, x_max, y_max = bbox
    bbox_width = max(0, x_max - x_min)
    bbox_height = max(0, y_max - y_min)
    bbox_area = bbox_width * bbox_height
    image_area = H * W
    return bbox_area / image_area

def crop_image(image,
               bbox,
               min_crop_ratio = 0.5,
               max_crop_ratio = 0.85,
               dynamic_margin = 1.5):
        
    H, W = image.shape[-2:]
    x0, y0, x1, y1 = bbox
 
    bbox_cx = (x0 + x1) // 2 
    bbox_cy = (y0 + y1) // 2 
    bbox_w = x1 - x0 
    bbox_h = y1 - y0 
 
    # Expand bbox with margin 
    crop_w = int(bbox_w * dynamic_margin) 
    crop_h = int(bbox_h * dynamic_margin) 
 
    # Enforce a minimum crop size based on the image 
    min_crop_w = int(W * min_crop_ratio) 
    min_crop_h = int(H * min_crop_ratio) 
 
    crop_w = max(crop_w, min_crop_w) 
    crop_h = max(crop_h, min_crop_h) 
    
    # Enforce maximum crop size
    max_crop_w = int(W * max_crop_ratio)
    max_crop_h = int(H * max_crop_ratio)
    
    crop_w = min(crop_w, max_crop_w)
    crop_h = min(crop_h, max_crop_h)
 
    # Maintain aspect ratio 
    aspect = W / H 
    if crop_w / crop_h > aspect: 
        crop_h = int(crop_w / aspect) 
    else: 
        crop_w = int(crop_h * aspect) 
 
    # Compute crop boundaries 
    x_start = max(0, bbox_cx - crop_w // 2) 
    x_end = min(W, bbox_cx + crop_w // 2) 
    y_start = max(0, bbox_cy - crop_h // 2) 
    y_end = min(H, bbox_cy + crop_h // 2) 
 
    # Pad if crop is near edge 
    if x_end - x_start < crop_w: 
        pad = crop_w - (x_end - x_start) 
        x_start = max(0, x_start - pad // 2) 
        x_end = min(W, x_end + pad // 2) 
 
    if y_end - y_start < crop_h: 
        pad = crop_h - (y_end - y_start) 
        y_start = max(0, y_start - pad // 2) 
        y_end = min(H, y_end + pad // 2) 
    
    return image[:, y_start:y_end, x_start:x_end]

def resize_to_divisible(image, divisor=16, min_size=16):
    H, W = image.shape[-2:]
    new_H = max(min_size, math.ceil(H / divisor) * divisor)
    new_W = max(min_size, math.ceil(W / divisor) * divisor)
    if new_H != H or new_W != W:
        image = F.interpolate(image.unsqueeze(0),
                              size=(new_H, new_W),
                              mode='bilinear',
                              align_corners=False).squeeze(0)
    return image
