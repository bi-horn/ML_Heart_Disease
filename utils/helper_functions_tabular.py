#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:49:16 2024

@author: bibianamailyn
"""

import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.inception import InceptionOutputs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import roc_auc_score, roc_curve, auc

import os, time, random, torch, warnings
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score

#save the metrics after training
def save_metrics(loss, accuracy, validation_loss, validation_accuracy, model, save_path):
    np.save("{}{}_train_loss.npy".format(save_path, model), loss)
    np.save("{}{}_train_accuracy.npy".format(save_path, model), accuracy)
    np.save("{}{}_validation_loss.npy".format(save_path, model), validation_loss)
    np.save("{}{}_validation_accuracy.npy".format(save_path, model), validation_accuracy)

def train_model(model, X_train, y_train, X_test, y_test, model_name, save_path, optimizer, criterion, device, num_epochs, dtype):
    losses = []
    accuracies = []
    v_accuracies = []
    v_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        train_accuracy += torch.sum(preds == y_train.data)

        val_accuracy, val_loss, _, _ = evaluate_model(model, X_test, y_test, criterion, device, dtype)

        v_accuracies.append(val_accuracy)
        v_losses.append(val_loss)
        losses.append(train_loss)
        accuracies.append(train_accuracy.item() / len(X_train))

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy.item() / len(X_train)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

    save_metrics(losses, accuracies, v_losses, v_accuracies, model_name, save_path)

    current_time = time.time()
    total = current_time - start_time
    print(f'Training took: {total / 60} minutes')

    return losses, accuracies, v_accuracies, v_losses

def evaluate_model(model, X_test, y_test, criterion, device, dtype):
    model.eval()
    _loss, _pred, _true, _accuracy = 0.0, [], [], []

    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs, y_test)

        _loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        _pred.extend(predicted.cpu().numpy())
        _true.extend(y_test.cpu().numpy())

    _loss /= len(y_test)
    _accuracy = accuracy_score(_true, _pred)
    _recall = recall_score(_true, _pred, average='macro')
    _precision = precision_score(_true, _pred, average='macro')
    _fscore = f1_score(_true, _pred, average='macro')

    print('{}: Accuracy: {:.4f} | Loss: {:.4f} | Recall: {:.4f} | Precision: {:.4f} | F-score: {:.4f}'.format(dtype, _accuracy, _loss, _recall, _precision, _fscore))
    print("")

    return _accuracy, _loss, _true, _pred

