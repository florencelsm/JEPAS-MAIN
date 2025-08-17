from torch.utils.data import Dataset
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoFeatureExtractor
import torchvision.transforms as T
from dataset_utils import (load_audio, unnormalize_bbox, scale_bbox,
                                  get_bbox_ratio_img, crop_image,
                                  resize_to_divisible, resize_to_square, resize_with_aspect_ratio)
class VGGSS_Dataset(Dataset):
    def __init__(self,
                 audio_mode: str,
                 config: dict) -> None:
        self.config = config
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
        # if config["RESIZE"]:
        #     self.transform_resize = T.Resize((config["RESIZE"], config["RESIZE"]), 
        #                                      interpolation=T.InterpolationMode.BICUBIC)
        self.mode = config["MODE"]
        self._init_dataset(config["DATASET_PATH"])
    
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
        image = self.img_processor(Image.open(img_path), return_tensors="pt").pixel_values.squeeze(0)
        if self.config["RESIZE"]:
            if config["RESIZE_ASPECT_RATIO"]:
                image = resize_with_aspect_ratio(image, self.config["RESIZE"])
            else:
                image = resize_to_square(image, self.config["RESIZE"])
        if self.config["RETURN_BBOX"]:
            bounding_box = []
            for box in bbox:
                box = unnormalize_bbox(box, original_size)
                box = scale_bbox(box, original_size, image.shape[-2:])
                bounding_box.append(box)
            return {"waveform": waveform, "image": image, "bbox": bounding_box, "original_size": original_size}
        else:
            return {"waveform": waveform, "image": image}

if __name__ == "__main__":
    config = {"DATASET_PATH": "/home/ec2-user/vggss/JEPAS-MAIN/vggss_data_clean/clean_extracted_data.json",
                 "MODE": "train",
                 "SHUFFLE_DATASET": True,
                 "DO_CENTER_CROP": False,
                 "RESIZE": 448,
                 "RESIZE_ASPECT_RATIO": False,
                 "RETURN_BBOX": False,
                 "RESIZE_DINO": False,
                 "CROP_AT_BBOX": False,
                 "RATIO_BBOX": 0.7,
                 "MIN_CROP_RATIO": 0.5,
                 "MAX_CROP_RATIO": 0.85,
                 "MIN_ABS_SIZE": 144, 
                 "DYNAMIC_MARGIN": 3.0,
                 "SAMPLE_RATE": 16000,
                 "MIN_WAVEFORM_LEN": 400}
    dataset = VGGSS_Dataset(audio_mode="waveform", config=config)
    import cv2
    from torch.utils.data import DataLoader
    import numpy as np
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    x = 23
    for i, batch in enumerate(dataloader):
        if i <= x:
            continue
        # Convert tensor to NumPy array
        img_tensor = batch['image'].squeeze().cpu()  # Remove batch dimension and move to CPU

        # Permute from (C, H, W) to (H, W, C) for OpenCV
        img_np = img_tensor.permute(1, 2, 0).numpy()

        # Convert data type and channel order for OpenCV (uint8, BGR)
        img_bgr = (img_np)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        bbox = batch['bbox']
        for box in bbox:
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0.item()), int(y0.item()), int(x1.item()), int(y1.item())
            cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (255, 0, 0), 2)

        # Save the image
        cv2.imwrite(f"output_image_{i}.png", img_bgr)
        if i == x+5:
            break
    print("Data loading complete.")