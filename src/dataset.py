import os
from PIL import Image
from torch.utils.data import Dataset


class PaddyDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, image_files, image_transform=None, mask_transform=None):
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        mask_name = os.path.splitext(image_name)[0] + ".png"

        image_path = os.path.join(self.rgb_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()

        return image, mask
