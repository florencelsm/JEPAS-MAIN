from torch.utils.data import Dataset
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoFeatureExtractor
from dataset.dataset_utils import load_audio, unnormalize_bbox, scale_bbox, get_bbox_ratio_img, crop_image

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
        if not config["RESIZE"]:
            self.img_processor.do_resize = False
        else:
            self.img_processor.size["shortest_edge"] = config["RESIZE"]
        self.mode = config["MODE"]
        self._init_dataset(config["DATASET_PATH"])
    
    def _init_dataset(self, dataset_path: str):
        with open(dataset_path, "r") as f:
            raw_data = json.load(f)
        self.items = []
        for _, data in raw_data.items():
            audio_path, image_path, bboxes, original_size = data
            self.items.append((audio_path, image_path, bboxes[0], original_size))

    def __len__(self) -> int: 
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        wave_path, img_path, bbox, original_size = self.items[idx]
        waveform, sample_rate = load_audio(wave_path, self.audio_processor.sampling_rate)
        waveform = self.audio_processor(waveform,
                                        sampling_rate=sample_rate, 
                                        return_tensors="pt").input_values.squeeze(0)
        image = self.img_processor(Image.open(img_path), return_tensors="pt").pixel_values.squeeze(0)
        if self.config["CROP_AT_BBOX"] and self.mode == 'train':
            bbox = unnormalize_bbox(bbox, original_size)
            bbox = scale_bbox(bbox, original_size, image.shape[-2:])
            ratio_bbox = get_bbox_ratio_img(bbox, image.shape[-2:])
            if ratio_bbox < self.config["RATIO_BBOX"]:
                image = crop_image(image,
                                   bbox,
                                   self.config["MIN_CROP_RATIO"],
                                   self.config["MAX_CROP_RATIO"],
                                   self.config["DYNAMIC_MARGIN"],)
        return {"waveform": waveform, "image": image}