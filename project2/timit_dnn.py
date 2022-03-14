# Train a torch DNN for Kaldi DNN-HMM model

import math
import sys

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2
HIDDEN_DIM = 256
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]

TRAIN_ALIGNMENT_DIR = "exp/tri_ali_train"
DEV_ALIGNMENT_DIR = "exp/tri_ali_dev"
TEST_ALIGNMENT_DIR = "exp/tri_ali_test"


def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    avg_train_losses = []  # track train loss in each epoch
    avg_val_losses = []  # track validation loss in each epoch
    min_val_loss = np.Inf
    epochs_no_improve = 0

    # train model
    model.train()
    for epoch in range(1, epochs + 1):
        train_losses = []
        val_losses = []
        for i, data in enumerate(train_loader):
            X_batch, y_batch= data
            optimizer.zero_grad()
            out = model(X_batch.float())
            loss = criterion(out, y_batch.long())
            loss.backward()
            optimizer.step()
            # track loss
            train_losses.append(loss.detach().item())
        avg_train_loss = np.average(train_losses)
        avg_train_losses.append(avg_train_loss)

        # calculate validation loss
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                X_batch, y_batch= data
                out = model(X_batch.float())
                loss = criterion(out, y_batch.long())  # loss
                val_losses.append(loss.detach().item())
        avg_val_loss = np.average(val_losses)
        avg_val_losses.append(avg_val_loss)

        # print information
        print("Epoch: {}  -  loss: {}  -  val_loss: {}".format(epoch, avg_train_loss, avg_val_loss))

        # early stopping
        if avg_val_loss < min_val_loss:
            torch.save(model, BEST_CHECKPOINT)  # save checkpoint
            epochs_no_improve = 0
            min_val_loss = avg_val_loss
        else:
            epochs_no_improve += 1
        if epoch > patience and epochs_no_improve == patience:
            print('Early stopping')
            break


trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')

scaler = StandardScaler()
scaler.fit(trainset.feats)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)


model = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
model.to(DEVICE)

print(f"The network architecture is: \n {model}")

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train(model, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
