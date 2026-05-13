import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.cli import load_stacked_flows


class PalmDataset(Dataset):
    def __init__(self, crease_root, palm_root=None, flow_path=None, per_ID_num=1):
        """
        crease_root: path to crease images (required).
        palm_root: path to palm images (optional; None if absent).
        flow_path: precomputed flows (.npz with key 'flows' or legacy .npy); optional.
        """
        self.per_ID_num = per_ID_num
        self.crease_paths = sorted(self._make_dataset(crease_root) * per_ID_num)
        self.palm_paths = sorted(self._make_dataset(palm_root)) if palm_root is not None else None

        if flow_path is not None:
            self.flow_list = torch.from_numpy(load_stacked_flows(flow_path)).float()
        else:
            self.flow_list = None

        print(f"[INFO] Crease dataset: {len(self.crease_paths)//per_ID_num} samples.")
        if self.palm_paths is not None:
            print(f"[INFO] Palm dataset: {len(self.palm_paths)} samples.")
        else:
            print(f"[INFO] Palm dataset: None")

        if self.flow_list is not None:
            print(f"[INFO] Flow dataset: {len(self.flow_list)} samples.")
        else:
            print(f"[INFO] Flow dataset: None")

        self.crease_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # [-1,1]
        ])

        self.palm_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
        ])

    def __getitem__(self, index):
        crease_path = self.crease_paths[index % len(self.crease_paths)]
        crease = self.crease_transform(Image.open(crease_path).convert("L"))

        # Random palm image if available, else None
        if self.palm_paths is not None:
            palm_path = self.palm_paths[random.randint(0, len(self.palm_paths) - 1)]
            palm = self.palm_transform(Image.open(palm_path).convert("RGB"))
        else:
            palm = None

        # Random flow sample if available, else None
        if self.flow_list is not None:
            flow = self.flow_list[random.randint(0, len(self.flow_list) - 1)]
        else:
            flow = None

        return {
            "crease": crease,   # [1, 256, 256]
            # "palm": palm,       # [3, 256, 256] or None
            # "flow": flow        # [2, H, W] or None
        }

    def __len__(self):
        return len(self.crease_paths)

    def _make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)