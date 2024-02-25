#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Neural network

13 features as input and 2 classes as output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  #input layer with 13 features (columns in df)
  #2 hidden layers with neurons
  #output layer with 2 classes of target variable (Risk/No risk)

  def __init__(self, in_features=13, h1=26, h2=30, out_features=2):
    super(Model, self).__init__()
    #super().__init__() # instantiate nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    return x
