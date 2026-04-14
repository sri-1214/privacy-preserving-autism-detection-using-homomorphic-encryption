import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import random


class MRIDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        autistic = []
        control = []

        classes = {"Autistic": 1, "Typical_Control": 0}

        # Load paths
        for label_name, label in classes.items():
            folder = os.path.join(root_dir, label_name)

            for file in os.listdir(folder):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    path = os.path.join(folder, file)

                    if label == 1:
                        autistic.append((path, label))
                    else:
                        control.append((path, label))

        random.shuffle(autistic)
        random.shuffle(control)

        # 🔥 Balanced training set (1500 total → 750 each)
        n = min(len(autistic), len(control), 750)

        train_autistic = autistic[:n]
        train_control = control[:n]

        # Remaining = test set
        test_autistic = autistic[n:]
        test_control = control[n:]

        if split == "train":
            self.samples = train_autistic + train_control
        else:
            self.samples = test_autistic + test_control

        random.shuffle(self.samples)

        print(f"\n[{split.upper()} SET]")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = nib.load(path).get_fdata()

        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)

        # 4D → 3D
        if img.ndim == 4:
            img = img[..., 0]

        # 2D → fake 3D
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 48, axis=-1)

        if img.ndim != 3:
            raise ValueError(f"Invalid shape {img.shape} in {path}")

        # Resize
        img = self.resize_volume(img, (48, 48, 48))

        # 🔥 Augmentation (only for training)
        if random.random() > 0.5:
            img = np.flip(img, axis=0)

        if random.random() > 0.5:
            img = np.flip(img, axis=1)

        # Fix memory
        img = np.ascontiguousarray(img)

        img = np.expand_dims(img, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label)

    def resize_volume(self, img, target_shape):
        factors = [t / s for t, s in zip(target_shape, img.shape)]
        return zoom(img, factors)