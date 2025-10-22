import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from sklearn.metrics import f1_score 
import time

class MLPmodel(nn.Module):
    def __init__(self,input_size,output_size,embed_size):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size,embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(embed_size,output_size)
    
    def forward(self,x): 
        x = self.sequential(x)
        output = self.classifier(x)

        return output
    

def plot(fig, ax1, ax2, train_loss, test_loss, f1_train, f1_test):
    ax1.clear()
    ax1.plot(train_loss, label = "Training Loss")
    ax1.plot(test_loss, label = "Testing Loss")
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.clear()
    ax2.plot(f1_train, label = "Training F1")
    ax2.plot(f1_test, label = "Testing F1")
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1_Score')
    ax2.legend()

    fig.tight_layout()
    clear_output(wait=True)
    display(fig)
    time.sleep(0.1) 

    return

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs =500):
    train_losses = []
    test_losses = []
    train_f1 = []
    test_f1 = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    display(fig)

    
    for epoch in range(num_epochs):
        model.train()
        current_train_loss = 0.0
        for i, (features, labels) in enumerate(train_loader):
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_train_loss += loss.item()
        train_losses.append(current_train_loss / len(train_loader))

        model.eval()
        current_test_loss = 0.0
        all_train_preds, all_train_labels = [], []
        all_test_preds, all_test_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                current_test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_test_preds.append(preds.numpy())
                all_test_labels.append(labels.numpy())
            
            for inputs, labels in train_loader:
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_train_preds.append(preds.numpy())
                all_train_labels.append(labels.numpy())
            
        test_losses.append(current_test_loss / len(test_loader))

        train_f1.append(f1_score(np.vstack(all_train_labels), np.vstack(all_train_preds), average='micro'))
        test_f1.append(f1_score(np.vstack(all_test_labels), np.vstack(all_test_preds), average='micro'))

        plot(fig, ax1, ax2,train_losses, test_losses, train_f1, test_f1)

