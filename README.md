# Multimodal COVID-19 Diagnosis using Vision and Language

This project combines vision (chest X-ray images) and text (clinical symptoms) to classify COVID-19 and pneumonia using a multimodal deep learning model built in PyTorch.

## Project Structure

```
project/
│
├── data/
│   └── chest_xray/                # Chest X-ray dataset from Kaggle
│       ├── train/
│       ├── test/
│       └── val/
│
├── models/
│   └── multimodal_model.py        # PyTorch model definition
│
├── utils/
│   └── dataset.py                 # Dataset and preprocessing logic
│
├── train.py                       # Training script
├── test.py                        # Testing script
├── requirements.txt
└── README.md                      # You're here
```

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset. Make sure you have the `kaggle` CLI installed and set up properly.

To download the dataset:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data
```

## Training

To train the model:

```bash
python train.py
```

### Important Notes:
- If you're using a weaker GPU (like GTX 1050), **adjust the training parameters accordingly** in `train.py`:
  - Reduce batch size (e.g., from 32 → 8)
  - Reduce image resolution
  - Lower the number of epochs
  - Optionally use CPU if CUDA is unavailable

Example adjustments:

```python
batch_size = 8
num_epochs = 5
learning_rate = 0.0001
```

## Testing

Make sure you have a trained model saved as `best_model.pth` in the project root or specify the correct path.

```bash
python test.py --model_path best_model.pth
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Model Architecture

The model uses:

- A CNN backbone (e.g., ResNet18) for image encoding
- An LSTM or transformer-based encoder for symptom text
- A classifier that combines both modalities

## Tips

- If you're modifying the model or using a different number of classes, **ensure the final layer in `classifier` matches your dataset.**
- Hyperparameters are defined in `train.py` — feel free to experiment.
- You can optionally create a metadata CSV with symptoms and image paths for true multimodal training.

---

**Disclaimer**: This model is for academic and experimental use only. Do not use for actual clinical diagnosis.
