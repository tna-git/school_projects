"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        self.linear1 = nn.Linear(indim, hidden_layer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_layer, outdim)


    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length
model = None
criterion = None

def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).long()
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
    return avg_loss, accuracy

if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """


    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500

    #dataset
    Xtrain = pd.read_csv("C:/Users/Trish/Documents/CMU/NN/HW1/HW1/data/X_train.csv")
    Ytrain = pd.read_csv("C:/Users/Trish/Documents/CMU/NN/HW1/HW1/data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("C:/Users/Trish/Documents/CMU/NN/HW1/HW1/data/X_test.csv")
    Ytest = pd.read_csv("C:/Users/Trish/Documents/CMU/NN/HW1/HW1/data/y_test.csv")
    Xtest = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest).to_numpy()
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #construct the training process
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Construct the training process
    print("\nStarting training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            epoch_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_train_total += y_batch.size(0)
            epoch_train_correct += (predicted == y_batch).sum().item()
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_correct / epoch_train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        
        # Validation
        avg_test_loss, avg_test_acc = validate(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_acc)
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={avg_train_acc:.4f}, Test Loss={avg_test_loss:.4f}, Test Acc={avg_test_acc:.4f}")
    
    
    # Plotting
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(test_accuracies, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('reference_results.png')
    plt.show()
    
    # Print final results
    print(f"\nFinal Results:")
    print(f"Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}")
