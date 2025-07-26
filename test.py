import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import DistilBertTokenizerFast
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

TEMPLATES = {
    'PNEUMONIA': [
        "Patient shows signs of pneumonia with lung infiltrates.",
        "Chest X-ray indicates infection consistent with pneumonia.",
        "Radiograph reveals lung opacity consistent with pneumonia."
    ],
    'NORMAL': [
        "No signs of infection or abnormal opacity. Lungs are clear.",
        "Normal chest X-ray with no pathologic findings.",
        "Patient exhibits no signs of distress; radiograph is unremarkable."
    ]
}

class TestMultimodalDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transform=None, max_length=128):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'test'].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = row['label']

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        synthetic_note = random.choice(TEMPLATES[label])

        encoding = self.tokenizer(
            synthetic_note,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        label_map = {'NORMAL': 0, 'PNEUMONIA': 1}
        label_tensor = torch.tensor(label_map[label])

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label_tensor
        }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    test_dataset = TestMultimodalDataset(
        csv_file=os.path.join("data", "chest_xray_metadata.csv"),
        tokenizer=tokenizer,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    from models.multimodal_model import MultimodalCOVIDModel

    model = MultimodalCOVIDModel(text_input_dim=768, num_classes=2).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
