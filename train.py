"""
Fine-tunes a MobileNetV2 (pretrained on ImageNet) to classify recyclable waste
into 6 categories, then exports it to ONNX for lightweight deployment.

Run once, locally: python train.py
Produces: model/waste_classifier.onnx, model/labels.json
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

DATASET_ID = "garythung/trashnet"
IMAGES_PER_CLASS = 150
IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3
MODEL_DIR = Path(__file__).parent / "model"

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class WasteDataset(Dataset):
    def __init__(self, examples, label_names):
        self.examples = examples
        self.label_names = label_names

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example["image"].convert("RGB")
        label = example["label"]
        return TRAIN_TRANSFORM(image), label


def build_balanced_subset():
    """Streams the dataset and keeps up to IMAGES_PER_CLASS examples per label,
    so we don't have to download the full ~3.7GB dataset."""
    print(f"Streaming {DATASET_ID} and sampling {IMAGES_PER_CLASS} images/class...")
    stream = load_dataset(DATASET_ID, split="train", streaming=True)
    label_names = stream.features["label"].names

    counts = {i: 0 for i in range(len(label_names))}
    examples = []
    for example in stream:
        label = example["label"]
        if counts[label] < IMAGES_PER_CLASS:
            examples.append(example)
            counts[label] += 1
        if all(c >= IMAGES_PER_CLASS for c in counts.values()):
            break

    print(f"Collected {len(examples)} images: {counts}")
    return examples, label_names


def build_model(num_classes):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train():
    examples, label_names = build_balanced_subset()
    dataset = WasteDataset(examples, label_names)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(len(label_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        total_loss, correct = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        avg_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {avg_loss:.4f} - acc: {accuracy:.2%}")

    export_onnx(model, label_names)


def export_onnx(model, label_names):
    MODEL_DIR.mkdir(exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    onnx_path = MODEL_DIR / "waste_classifier.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    (MODEL_DIR / "labels.json").write_text(json.dumps(label_names, indent=2))
    print(f"Exported {onnx_path} and labels.json")


if __name__ == "__main__":
    train()
