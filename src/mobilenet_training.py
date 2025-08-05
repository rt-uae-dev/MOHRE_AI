import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Config
load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "data" / "dataset"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "model_classifier.pt")
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 224

def train():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss:.4f}  Accuracy: {100 * correct / total:.2f}%")

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    start_time = datetime.now()
    train()
    print(f"Training completed in: {datetime.now() - start_time}")
