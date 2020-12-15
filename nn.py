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

TRAIN_DATA_SOURCE = "./data/train.csv"
TEST_DATA_SOURCE = "./data/test.csv"

train_data = pd.read_csv(TRAIN_DATA_SOURCE, quotechar='`')
test_data = pd.read_csv(TEST_DATA_SOURCE, quotechar='`')

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

# Loading data and splitting up words and masked words


def load_vocab_masks(text):
    word_to_ix = {}
    mask_to_ix = {}
    ix_to_mask = {}
    for mask, text in zip(train_data['mask'], train_data['text']):
        for word in text.split(' '):
            word_to_ix.setdefault(word, len(word_to_ix))
        mask_to_ix.setdefault(mask, len(mask_to_ix))
        ix_to_mask[mask_to_ix[mask]] = mask
    return word_to_ix, mask_to_ix, ix_to_mask


word_to_ix, mask_to_ix, ix_to_mask = load_vocab_masks(train_data)

# Initialize Model
emb_dim = 100
learning_rate = 0.01
model = Net(len(word_to_ix), emb_dim, len(mask_to_ix))
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

# Training
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    for mask, text in zip(train_data['mask'], train_data['text']):
        x = [word_to_ix[word] for word in text.split()]
        x_train_tensor = torch.LongTensor(x)
        x_train_tensor = torch.LongTensor(x)
        y_train_tensor = np.zeros(len(mask_to_ix.keys()))
        y_train_tensor[mask_to_ix[mask]] = 1
        y_train_tensor = torch.Tensor(y_train_tensor)
        pred_y = model(x_train_tensor)
        loss = loss_fn(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())

'''
Function for evaluating the model
@params
masks: list of potential masked words
texts: sample unfinished sentences with a missing word
'''


def evaluate(model, masks, texts):
    valid = 0
    total = 0
    top_k = 5
    for mask, text in zip(masks, texts):
        y_test = mask
        with torch.no_grad():
            model.eval()
            x = [word_to_ix[word]
                 for word in y_test.split() if word in word_to_ix]
            x_test = torch.LongTensor(x)
            pred_y_test = model(x_test)
            values, mask_indices = pred_y_test.topk(top_k)
            for i, mask_index in enumerate(mask_indices):
                pred_word = ix_to_mask[mask_index.item()]
                if mask == pred_word:
                    valid += 1
                    break
            total += 1
    print(f"Accuracy (k={top_k}): {valid/total}")


print("\Evaluate (NN): Test Data\n")
evaluate(model, test_data['mask'], test_data['text'])
