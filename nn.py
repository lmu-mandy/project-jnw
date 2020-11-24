import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import re

class Net(nn.Module):
    def __init__(self, num_words, emb_dim, num_y):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.linear = nn.Linear(emb_dim, num_y)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embeds = torch.mean(self.emb(text), dim=0)
        return self.sigmoid(self.linear(embeds))


def load_vocab(text):
    word_to_ix = {}
    for sent, _mask in text:
        for word in sent.split():
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix

# Data

df = pd.read_csv("data/train.csv", quotechar='`')
#   TODO: Store masks more efficiently
train_data = []
masks = []
for i, row in df.iterrows():
    if i != 0:
        text, mask = row['text'], row['mask']
        train_data.append((text,mask))
        if mask not in masks:
            masks.append(mask)
tok_to_ix = load_vocab(train_data)

# Model

emb_dim = 8
num_classes = len(masks)
learning_rate = 0.01
model = Net(len(tok_to_ix), emb_dim, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Training

n_epochs = 10
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

# Test

y_test = "I'm happy because of my"
with torch.no_grad():
  model.eval()
  x = [tok_to_ix[tok] for tok in y_test.split() if tok in tok_to_ix]
  x_test = torch.LongTensor(x)
  pred_y_test = model(x_test)
  print(pred_y_test)