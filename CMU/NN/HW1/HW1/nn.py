"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (indim, batch_size)
        """
        self.input = x
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        """
        grad_wrt_input = grad_wrt_out * (self.input > 0).float()
        return grad_wrt_input


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 *torch.rand((outdim, indim), dtype=torch.float64, requires_grad=False, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=False, device=device)
        self.lr = lr
        self.input = None
        self.grad_wrt_weights = None
        self.grad_wrt_bias = None

    def forward(self, x):
        """
        x shape (indim, batch_size)
        return shape (outdim, batch_size)
        """
        self.input = x
        return self.weights @ x + self.bias


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        return shape (indim, batch_size)
        """
        #compute grad_wrt_weights
        self.grad_wrt_weights = grad_wrt_out @ self.input.T
        #compute grad_wrt_bias
        self.grad_wrt_bias = grad_wrt_out.sum(dim=1, keepdim=True)
        #compute & return grad_wrt_input
        grad_wrt_input = self.weights.T @ grad_wrt_out
        return grad_wrt_input


    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        """
        if self.grad_wrt_weights is not None:
            self.weights -= self.lr * self.grad_wrt_weights
        if self.grad_wrt_bias is not None:
            self.bias -= self.lr * self.grad_wrt_bias


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        self.logits = logits
        self.labels = labels
        self.batch_size = logits.shape[1]
    
        logits_max = torch.max(logits, dim=0, keepdim=True)[0]
        exp_logits = torch.exp(logits - logits_max)
        self.softmax_probs = exp_logits / torch.sum(exp_logits, dim=0, keepdim=True)
        
        loss = -torch.sum(labels * torch.log(self.softmax_probs + 1e-10)) / self.batch_size
        
        return loss

    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        grad_wrt_logistics = (self.softmax_probs - self.labels) / self.batch_size
        return grad_wrt_logistics
    
    def getAccu(self):
        """
        return accuracy here
        """
        predicted_classes = torch.argmax(self.logits, dim=0)
        true_classes = torch.argmax(self.labels, dim=0)
        accuracy = torch.sum(predicted_classes == true_classes).item() / self.batch_size
        return accuracy


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.linear1 = LinearMap(indim, hidden_layer, lr=lr)
        self.relu = ReLU()
        self.linear2 = LinearMap(hidden_layer, outdim, lr=lr)


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        out = self.linear1.forward(x)
        out = self.relu.forward(out)
        logi = self.linear2.forward(out)
        return logi


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad = self.linear2.backward(grad_wrt_out)
        grad = self.relu.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

    
    def step(self):
        """update model parameters"""
        self.linear1.step()
        self.linear2.step()


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

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

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
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim=indim, outdim=outdim, hidden_layer=hidden_dim, lr=lr)
    criterion = SoftmaxCrossEntropyLoss()
    #construct the training process
    train_loss = []
    train_accu = []
    test_loss = []
    test_accu = []

    for epoch in range(epochs):
        # Training
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_batches = 0
    
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.T.to(device).double()
            y_batch = y_batch.numpy()
            y_onehot = labels2onehot(y_batch).T
            y_onehot = torch.tensor(y_onehot, dtype=torch.float64, device=device)
            
            logits = model.forward(x_batch)
            loss = criterion.forward(logits, y_onehot)
            grad = criterion.backward()
            model.backward(grad)
            model.step()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += criterion.getAccu()
            num_train_batches += 1
        
        train_loss.append(epoch_train_loss / num_train_batches)
        train_accu.append(epoch_train_acc / num_train_batches)
        
        # Testing
        epoch_test_loss = 0.0
        epoch_test_acc = 0.0
        num_test_batches = 0
        
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.T.to(device).double()
            y_batch = y_batch.numpy()
            y_onehot = labels2onehot(y_batch).T
            y_onehot = torch.tensor(y_onehot, dtype=torch.float64, device=device)
            
            logits = model.forward(x_batch)
            loss = criterion.forward(logits, y_onehot)
            
            epoch_test_loss += loss.item()
            epoch_test_acc += criterion.getAccu()
            num_test_batches += 1
        
        test_loss.append(epoch_test_loss / num_test_batches)
        test_accu.append(epoch_test_acc / num_test_batches)
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}: Train Loss={train_loss[-1]:.4f}, Train Acc={train_accu[-1]:.4f}, Test Loss={test_loss[-1]:.4f}, Test Acc={test_accu[-1]:.4f}")
    
    # Plotting
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_loss, label='Train Loss')
    ax1.plot(test_loss, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accu, label='Train Accuracy')
    ax2.plot(test_accu, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

