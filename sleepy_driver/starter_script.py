import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

'''
ML FOR SIMPLE EEG DATA
Starter Code
Everything up to preprocessing is done already so you don't have to do it again.
'''

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load data and split accordingly
# we're gonna ignore subject splitting here since the dataset is vague about that
data_raw = pd.read_csv("data/acquiredDataset.csv")
datas0 = data_raw[data_raw['classification'] == 0]
datas1 = data_raw[data_raw['classification'] == 1]

# create windows of size 5
def create_windows(data, window_size=5):
    windows = []
    labels = []
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i + window_size].drop('classification', axis=1)
        label = data.iloc[i + window_size - 1]['classification']
        windows.append(np.array(window))
        labels.append(label)
    return windows, labels

w0, l0 = create_windows(datas0)
w1, l1 = create_windows(datas1)

# concatenate lists
w = np.array(w0 + w1)
l = l0 + l1

# split into 80/10/10 train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(w, l, test_size=0.2, random_state=42, stratify=l)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# normalize data using StandardScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_valid = scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

'''
All you now
'''

# create a pytorch dataset class that takes the windows and labels as parameters 
class EEGDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

# create dataloaders

# construct a simple model for eval
class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

# train and validate function
def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, epochs):
    pass
# train_and_validate(...)

# build evaluation function
def evaluate_model(model, dataloader, criterion):
  model.eval()
  loss = 0.0
  correct = 0
  total = 0

  predlist = []
  ylist = []

  for i, (X,y) in enumerate(dataloader):
    with torch.inference_mode():
      pass

  return {
      'model_name' : type(model).__name__,
      'loss' : loss,
      'acc' : round(correct / total,4),
    }, classification_report(ylist, predlist)

# test_results, report = evaluate_model(model, test_loader, criterion)