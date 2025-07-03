from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torchvision.transforms as T
import os

base_path = "./data/image/sample_imagenet"

for split in ["train", "val", "test"]:
    for class_id in range(3):  # 模拟3个类别
        os.makedirs(f"{base_path}/{split}/class{class_id}", exist_ok=True)

transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
dataset = CIFAR10(root=".", train=True, download=True, transform=transform)

for class_id in range(3):
    imgs = [img for img, label in dataset if label == class_id][:10]
    for i, img in enumerate(imgs):
        for split in ["train", "val", "test"]:
            save_image(img, f"{base_path}/{split}/class{class_id}/img_{i:02d}.jpg")
