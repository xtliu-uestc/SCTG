from typing import Callable, Optional, Any, List
from torchvision.datasets import CIFAR100, CIFAR10, ImageFolder
from torch.utils.data import Dataset, ConcatDataset
import random
import numpy as np
import json
import os
from util.utils import pil_loader, remove_special_chars
from torchvision import transforms
from PIL import Image

class PACSDataset(Dataset):
    """PACS single-domain dataset, fully aligned with the DomainNetDataset interface.
    Split logic: first, a global 9:1 split separates test and trainval pools; the trainval pool is then further split 9:1 into final train/val via stratified sampling.
    """
    def __init__(self, root, domain_name, split='train', 
                 val_ratio=0.1, 
                 transform=None, random_seed=42):
        self.domain_name = domain_name
        self.transform = transform
        CLASS_NAMES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

        # test directly uses the global fixed test set; train/val are stratified from the trainval pool (xxx_train.txt).
        txt_split = 'train' if split in ['train', 'val'] else 'test'
        txt_path = os.path.join(root, 'pacs', f"{domain_name}_{txt_split}.txt")

        all_samples = []
        with open(txt_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label_str = line.rsplit(' ', 1)
                label = int(label_str)
                full_img_path = os.path.join(root, 'PACS', rel_path)
                all_samples.append((full_img_path, label))

        if split == 'test':
            self.samples = all_samples
        else:
            # Within the training pool, stratified 9:1 split into train/val by class.
            random.seed(random_seed)
            class_indices = {lbl: [] for lbl in self.class_to_idx.values()}
            for idx, (_, label) in enumerate(all_samples):
                class_indices[label].append(idx)

            selected_indices = []
            for label, indices in class_indices.items():
                random.shuffle(indices)
                n_total = len(indices)
                n_val = int(n_total * val_ratio)
                if split == 'val':
                    selected_indices.extend(indices[:n_val])
                else:
                    selected_indices.extend(indices[n_val:])
            self.samples = [all_samples[i] for i in selected_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = pil_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label, idx


def get_pacs_merged_dataset(root, split='train', domains=None, **kwargs):
    """Merge multi-domain PACS datasets, fully unified with the get_domainnet_merged_dataset calling convention."""
    if domains is None:
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    domain_datasets = []
    for domain in domains:
        ds = PACSDataset(root, domain, split=split,** kwargs)
        domain_datasets.append(ds)
    merged_dataset = ConcatDataset(domain_datasets)
    merged_dataset.classes = domain_datasets[0].classes
    return merged_dataset


OFFICEHOME_CLASS_NAMES =[
    "Alarm_Clock", "Backpack", "Batteries", "Bed", "Bike", "Bottle", "Bucket",
    "Calculator", "Calendar", "Candles", "Chair", "Clipboards", "Computer", "Couch",
    "Curtains", "Desk_Lamp", "Drill", "Eraser", "Exit_Sign", "Fan", "File_Cabinet",
    "Flipflops", "Flowers", "Folder", "Fork", "Glasses", "Hammer", "Helmet", "Kettle",
    "Keyboard", "Knives", "Lamp_Shade", "Laptop", "Marker", "Monitor", "Mop", "Mouse",
    "Mug", "Notebook", "Oven", "Pan", "Paper_Clip", "Pen", "Pencil", "Postit_Notes",
    "Printer", "Push_Pin", "Radio", "Refrigerator", "Ruler", "Scissors", "Screwdriver",
    "Shelf", "Sink", "Sneakers", "Soda", "Speaker", "Spoon", "Table", "Telephone",
    "ToothBrush", "Toys", "Trash_Can", "TV", "Webcam"
]


class OfficeHomeDataset(Dataset):
    """OfficeHome single-domain dataset, with interface fully consistent with PACS/DomainNet.
    Split logic: stratified 9:1 split into train/val by class within the trainval pool; test uses the fixed test set.
    """
    def __init__(self, root, domain_name, split='train',
                 val_ratio=0.1,
                 transform=None, random_seed=42):
        self.domain_name = domain_name
        self.transform = transform
        self.classes = OFFICEHOME_CLASS_NAMES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        txt_split = 'train' if split in ['train', 'val'] else 'test'
        txt_path = os.path.join(root, 'officehome', f"{domain_name}_{txt_split}.txt")

        all_samples = []
        with open(txt_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label_str = line.rsplit(' ', 1)
                label = int(label_str)
                full_img_path = os.path.join(root, 'OfficeHome', rel_path)
                all_samples.append((full_img_path, label))

        if split == 'test':
            self.samples = all_samples
        else:
            random.seed(random_seed)
            class_indices = {lbl: [] for lbl in self.class_to_idx.values()}
            for idx, (_, label) in enumerate(all_samples):
                class_indices[label].append(idx)

            selected_indices = []
            for label, indices in class_indices.items():
                random.shuffle(indices)
                n_val = int(len(indices) * val_ratio)
                if split == 'val':
                    selected_indices.extend(indices[:n_val])
                else:
                    selected_indices.extend(indices[n_val:])
            self.samples = [all_samples[i] for i in selected_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = pil_loader(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label, idx


def get_officehome_merged_dataset(root, split='train', domains=None, **kwargs):
    """Merge multi-domain OfficeHome datasets."""
    if domains is None:
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    domain_datasets = []
    for domain in domains:
        ds = OfficeHomeDataset(root, domain, split=split, **kwargs)
        domain_datasets.append(ds)
    merged_dataset = ConcatDataset(domain_datasets)
    merged_dataset.classes = domain_datasets[0].classes
    return merged_dataset