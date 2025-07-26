import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
from transformers import DistilBertTokenizerFast
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from models.multimodal_model import MultimodalCOVIDModel

# Hyperparameters
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
TEXT_DIM = 768
NUM_CLASSES = 2  # Binary: NORMAL (0), PNEUMONIA (1)

def preprocess_tokenize(df, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    for text in df['synthetic_note']:
        enc = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
        input_ids.append(enc['input_ids'][0])
        attention_masks.append(enc['attention_mask'][0])
    df['input_ids'] = input_ids
    df['attention_mask'] = attention_masks
    return df

class PreTokenizedCXRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = {'NORMAL': 0, 'PNEUMONIA': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)

        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        label = torch.tensor(self.label_map[row['label']])

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv("data/dataset.csv")

    # Filter only classes we care about
    df = df[df['label'].isin(['NORMAL', 'PNEUMONIA'])].reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_df = preprocess_tokenize(train_df, tokenizer)
    val_df = preprocess_tokenize(val_df, tokenizer)

    train_dataset = PreTokenizedCXRDataset(train_df, transform=transform)
    val_dataset = PreTokenizedCXRDataset(val_df, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    model = MultimodalCOVIDModel(text_input_dim=TEXT_DIM, num_classes=NUM_CLASSES).to(device)

    # Optional: Freeze encoder layers
    for param in model.cnn.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    log_path = "training_log.csv"
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        optimizer.zero_grad()
        for i, batch in loop:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            loop.set_postfix(loss=total_loss / (i + 1))

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total * 100

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, acc])

if __name__ == "__main__":
    train()
