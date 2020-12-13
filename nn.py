'''
Baseline Feed-Forward neural network with the purpose of identifying 
best preprocessing method for our project as well as providing a 
comparator to the GPT-2 language model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re

# Feed-Forward Network class constructor


class Net(nn.Module):
    def __init__(self, num_words, emb_dim, num_y):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.linear = nn.Linear(emb_dim, num_y)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embeds = torch.mean(self.emb(text), dim=0)
        return self.sigmoid(self.linear(embeds))


# Load data
def load_vocab(text):
    word_to_ix = {}
    for sent, _mask in text:
        for word in sent.split():
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix


# Function for loading training/testing data
def load_data(data_file):
    df = pd.read_csv(data_file, quotechar='`')
    dat = []
    masks = []
    for i, row in df.iterrows():
        if i != 0:
            text, mask = row['text'], row['mask']
            dat.append((text, mask))
            if mask not in masks:
                masks.append(mask)
    return dat, masks


# Load training data and split up built sentences and masks

train_data, masks = load_data("data/train.csv")
tok_to_ix = load_vocab(train_data)

# Load testing data and split up built sentences and masks

test_data, test_masks = load_data("data/test.csv")

# Initialize Model

emb_dim = 8
num_classes = len(masks)
learning_rate = 0.01
model = Net(len(tok_to_ix), emb_dim, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Train model on training data

n_epochs = 3
for epoch in range(n_epochs):
    model.train()
    for text, mask in train_data:
        x = [tok_to_ix[tok] for tok in text.split()]
        x_train_tensor = torch.LongTensor(x)
        y_train_tensor = np.zeros(len(masks))
        y_train_tensor[masks.index(mask)] = 1
        y_train_tensor = torch.Tensor(y_train_tensor)
        pred_y = model(x_train_tensor)
        loss = loss_fn(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())

# Manual Test: Uncomment to run your own manual test

# y_test = "I have the presentation to"

# with torch.no_grad():
#     model.eval()
#     x = [tok_to_ix[tok] for tok in y_test.split() if tok in tok_to_ix]
#     x_test = torch.LongTensor(x)
#     pred_y_test = model(x_test)
#     values, word_indices = pred_y_test.topk(3)
#     for i, word_index in enumerate(word_indices):
#         pred_word = masks[word_index]
#         print(values[i].item(), y_test, f"_{pred_word}_")


# Test Report
prediction_score = {"Yes": 0, "No": 0}
model.eval()
with torch.no_grad():
    for text, mask in test_data:
        y_test = mask
        x = [tok_to_ix[tok]
             for tok in y_test.split() if tok in tok_to_ix]
        x_test = torch.LongTensor(x)
        pred_y_test = model(x_test)
        res, ind = torch.topk(pred_y_test, 5)
        pred_words = set()
        for i in ind:
            pred_words.add(test_masks[i])
        if mask in pred_words:
            print("Correct Prediction:", pred_words, mask)
            prediction_score["Yes"] += 1
        else:
            print("Incorrect Prediction:", pred_words, mask)
            prediction_score["No"] += 1
    accuracy = prediction_score["Yes"] / \
        (prediction_score["Yes"] + prediction_score["No"])
    print("Correct Predictions (Topk | k=5):", prediction_score["Yes"])
    print("Incorrect Predictions (Topk | k=5):", prediction_score["No"])
    print("Accuracy:", accuracy)
