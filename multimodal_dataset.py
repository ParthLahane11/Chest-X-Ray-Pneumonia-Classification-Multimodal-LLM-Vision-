import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import os

class MultimodalCXRDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, transform=None):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.label_map = {label: idx for idx, label in enumerate(sorted(dataframe['label'].unique()))}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image = Image.open(row['image_path']).convert("RGB")
        image = self.transform(image)

        encoded = self.tokenizer(
            row['synthetic_note'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        label = self.label_map[row['label']]

        return {
            'image': image,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }


