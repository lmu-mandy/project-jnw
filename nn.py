import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from gensim.models import KeyedVectors


class FFNet(nn.Module):
    def __init__(self, num_words, emb_dim, num_y):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim)
        self.linear = nn.Linear(emb_dim, num_y)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text):
        embeds = self.emb(text)
        return self.softmax(self.linear(embeds))


# Check Word2Vec
# embeds = KeyedVectors.load_word2vec_format(
#     'GoogleNewsW2V.bin', binary=True).wv.vectors


def load_vocab(text):
    word_to_ix = {}
    for sent, label in text:
        for word in sent.split():
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix


train_data = []
df = pd.read_csv("emails.csv")
# Cols: 'file', 'message'
for index, row in df.iterrows():
    # if index > 20:
    #     break
    train_data.append((row['tweet'], row['class']))
tok_to_ix = load_vocab(train_data)

emb_dim = 300
num_classes = 3
learning_rate = 0.01
model = FFNet(len(tok_to_ix), emb_dim, num_classes, embeds)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    for text, label in train_data:
        x = [tok_to_ix[tok] for tok in text.split()]
        x_train_tensor = torch.LongTensor(x)
        # [2] -> [0, 0, 1]
        # [1] -> [0, 1, 0]
        # Changing length of the input for torch.Tensor to match what is expected based on number of classes
        list_input = []
        if label == 0:
            list_input = [0, 0, 0]
        elif label == 1:
            list_input = [0, 1, 0]
        elif label == 2:
            list_input = [0, 0, 1]
        y_train_tensor = torch.Tensor(list_input)
        pred_y = model(x_train_tensor)
        loss = loss_fn(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())
