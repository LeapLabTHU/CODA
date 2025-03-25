import os
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .utils import normalize_01_into_pm1

def build_imagenet_dataset(
    resolution=256,
    data_path=None,
    augmentation=False,
    split='train',
):
    mid_resolution = 1.125
    mid_resolution = round(mid_resolution * resolution)
    if augmentation:
        transform = transforms.Compose([
            transforms.Resize(mid_resolution, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
            transforms.RandomCrop((resolution, resolution)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ])


    root = os.path.join(data_path, split)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset
