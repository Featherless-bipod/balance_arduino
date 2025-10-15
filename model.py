import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class MLPmodel(nn.Module):
    def __init__(self,input_size,output_size,embed_size):
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



def train_model(model, criterion, optimizer, train_loader, test_loader num_epochs=500):
    train_losses = []
    test_losses = []
    train_f1_scores = []
    test_f1_scores = []

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
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_test_preds.append(preds.numpy())
                all_test_labels.append(labels.numpy())
            
            # Evaluate on training set (to compare train vs test performance)
            for inputs, labels in train_loader:
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_train_preds.append(preds.numpy())
                all_train_labels.append(labels.numpy())
            
        test_losses.append(current_test_loss / len(test_loader))