import torch
from torch.utils.data import Dataset
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoFeatureExtractor
import torchvision.transforms as T
from dataset.dataset_utils import (load_audio, unnormalize_bbox, scale_bbox,
                                  get_bbox_ratio_img, crop_image,
                                  resize_to_divisible, resize_to_square, resize_with_aspect_ratio)
class VGGSS_Dataset(Dataset):
    def __init__(self,
                 audio_mode: str,
                 mode: str,
                 config: dict) -> None:
        self.config = config
        self.audio_mode = audio_mode.lower()
        if audio_mode == "spectrogram":
            self.audio_processor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        elif audio_mode == "waveform":
            self.audio_processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        else:
            raise ValueError("audio_mode should be either spectrogram or waveform")
        self.img_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        if not config["DO_CENTER_CROP"]:
            self.img_processor.do_center_crop = False
        if not config["RESIZE_DINO"]:
            self.img_processor.do_resize = False
        else:
            self.img_processor.size["shortest_edge"] = config["RESIZE_DINO"]
        if config["RESIZE"]:
            self.transform_resize = T.Resize((config["RESIZE"], config["RESIZE"]), 
                                             interpolation=T.InterpolationMode.BICUBIC)
        self.mode = mode
        if mode == "train":
            dataset_path = config["TRAIN_DATASET_PATH"]
        else:
            dataset_path = config["TEST_DATASET_PATH"]
        self._init_dataset(dataset_path)
    
    def _init_dataset(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            raw_data = json.load(f)
        self.items = []
        for _, data in raw_data.items():
            images, audio, bbox, original_size = data
            for img in images:
                item = [img, audio, bbox, original_size]
                self.items.append(item)

    def __len__(self) -> int: 
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, wave_path, bbox, original_size = self.items[idx]
        waveform, sample_rate = load_audio(wave_path, self.audio_processor.sampling_rate)
        waveform = self.audio_processor(waveform,
                                        sampling_rate=sample_rate, 
                                        return_tensors="pt").input_values.squeeze(0)
        if self.audio_mode == "waveform":
            min_audio_length = 80000
            if waveform.shape[0] < min_audio_length:
                waveform = torch.nn.functional.pad(waveform, (0, min_audio_length - waveform.shape[0]), "constant", 0) 
            if waveform.shape[0] > min_audio_length:
                waveform = waveform[:min_audio_length]
            
        image = self.img_processor(Image.open(img_path), return_tensors="pt").pixel_values.squeeze(0)
        if self.config["RESIZE"]:
            if self.config["RESIZE_ASPECT_RATIO"]:
                image = resize_with_aspect_ratio(image, self.config["RESIZE"])
            else:
                image = resize_to_square(image, self.config["RESIZE"])
        if self.config["RETURN_BBOX"]:
            bounding_box = []
            for box in bbox:
                bounding_box.append(box)
            original_size = torch.as_tensor([original_size[1], original_size[0]], dtype=torch.float32)
            return {"waveform": waveform, "image": image, "bbox": bounding_box, "original_size": original_size}
        else:
            return {"waveform": waveform, "image": image}
    
    def collate_fn(self, batch):
        waveform = torch.stack([item["waveform"] for item in batch])
        image = torch.stack([item["image"] for item in batch])
        if self.config["RETURN_BBOX"]:
            bbox = [torch.as_tensor(item["bbox"], dtype=torch.float32) for item in batch]
            original_size = torch.stack([item["original_size"] for item in batch])
            return {"waveform": waveform, "image": image, "bbox": bbox, "original_size": original_size}
        else:
            return {"waveform": waveform, "image": image}

if __name__ == "__main__":
    config = {"TRAIN_DATASET_PATH": "/home/ec2-user/vggss/JEPAS-MAIN/vggss_data_clean/clean_extracted_data_train.json",
                 "MODE": "train",
                 "SHUFFLE_DATASET": True,
                 "DO_CENTER_CROP": False,
                 "RESIZE": 448,
                 "RESIZE_ASPECT_RATIO": False,
                 "RETURN_BBOX": True,
                 "RESIZE_DINO": False,
                 "CROP_AT_BBOX": False,
                 "RATIO_BBOX": 0.7,
                 "MIN_CROP_RATIO": 0.5,
                 "MAX_CROP_RATIO": 0.85,
                 "MIN_ABS_SIZE": 144, 
                 "DYNAMIC_MARGIN": 3.0,
                 "SAMPLE_RATE": 16000,
                 "MIN_WAVEFORM_LEN": 400}
    dataset = VGGSS_Dataset(audio_mode="spectrogram", mode='train', config=config)
    import cv2
    from torch.utils.data import DataLoader
    import numpy as np
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    x = 1
    for i, batch in enumerate(dataloader):
        if i <= x:
            continue
        # Convert tensor to NumPy array
        img_tensor = batch['image'].squeeze().cpu()  # Remove batch dimension and move to CPU

        # Permute from (C, H, W) to (H, W, C) for OpenCV
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Convert data type and channel order for OpenCV (uint8, BGR)
        img_bgr = (img_np)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR) * 255
        H, W = img_bgr.shape[:2]

        bbox = batch['bbox']
        for box in bbox:
            for i, b in enumerate(box):
                x0, y0, x1, y1 = b
                x0_px, y0_px, x1_px, y1_px = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
                cv2.rectangle(img_bgr, (x0_px, y0_px), (x1_px, y1_px), (0, 255, 0), 2)

        # Save the image
        cv2.imwrite(f"output_image_{i}.png", img_bgr)
        break
    # print("Data loading complete.")