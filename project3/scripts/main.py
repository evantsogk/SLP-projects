import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.25d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 25
EMB_TRAINABLE = False
BATCH_SIZE = 32
EPOCHS = 50
#DATASET = "MR"  # options: "MR", "Semeval2017A"
DATASET = "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
print("\nFirst 10 labels of training data:\n", y_train[:10])
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size
print("\nFirst 10 labels of training data, encoded:\n", y_train[:10])

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

print("\nFirst 10 training examples after preprocessing:")
for i in range(10):
    print(train_set.data[i])

# find a good length for sentences
plt.figure()
plt.scatter(list(range(len(train_set.data))), [len(text) for text in train_set.data])
plt.xlabel("Text index")
plt.ylabel("Text length")
plt.title("Text lengths of " + DATASET + " dataset")

for i in range(5):
    print("\ndataitem =", train_set.data[i])
    print("Return values:\nexample =", train_set[i][0], "\nlabel =", train_set[i][1], "\nlength =", train_set[i][2])


# EX4 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)  # EX7


#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

criterion = nn.CrossEntropyLoss()  # EX8
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
#print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = [p for p in model.parameters() if p.requires_grad]  # EX8
optimizer = optim.Adam(parameters, lr=0.001)  # EX8


#############################################################################
# Training Pipeline
#############################################################################
train_losses = []  # all train losses for plot
test_losses = []  # all test losses for plot
y_pred = None  # predicted
y_true = None  # true labels

PATIENCE = 5  # for early stopping
min_val_loss = np.Inf
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    y_pred = []  # predicted
    y_true = []  # true labels

    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    y_pred.extend(y_test_pred)
    y_true.extend(y_test_gold)

    # early stopping
    if test_loss < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = test_loss
    else:
        epochs_no_improve += 1
    if epoch > PATIENCE and epochs_no_improve == PATIENCE:
        print('Early stopping')
        break

print(classification_report(y_true, y_pred))

# train/test learning curves
plt.figure()
plot1, = plt.plot(train_losses, c='red')
plot2, = plt.plot(test_losses, c='green')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(" DNN Learning Curves (" + DATASET + " Dataset)")
plt.legend([plot1, plot2], ['train loss', 'test loss'])
plt.show()
