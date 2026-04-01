from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

print("Loading CIFAR-10...")
dataset = load_dataset("cifar10")
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Manual preprocessing - BATCHED correctly
def collate_fn(batch):
    images = [item["img"].convert("RGB") for item in batch]
    labels = torch.tensor([item["label"] for item in batch])
    inputs = processor(images, return_tensors="pt", padding=True)
    inputs["labels"] = labels
    return inputs

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(dataset["test"], batch_size=32, shuffle=False, collate_fn=collate_fn)

# Simple training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
model.train()

print("Training...")
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/3, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        pred = outputs.logits.argmax(-1)
        correct += (pred == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

print(f"Test Accuracy: {100*correct/total:.2f}%")