import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools
import random
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
    for sent, label in text:
        for word in sent.split():
            word_to_ix.setdefault(word, len(word_to_ix))
    return word_to_ix


train_data = []
df = pd.read_csv("emails.csv", nrows=500)
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))
# tok_to_ix = load_vocab(train_data)

# updated_data = train_data
train_data[:] = [' '.join(x).split('. ') for x in train_data] # i'll fix all this later lol too tired rn
train_data[:] = [[elem for elem in x if elem.strip()] for x in train_data]
train_data[:] = [[elem.strip() for elem in x] for x in train_data]

train_data = list(itertools.chain.from_iterable(train_data))

rand_word = random.choice(train_data[0].split())

# start_index = train_data[0].find(rand_word)
# end_index = start_index + len(rand_word)
# print(rand_word)
# print(train_data[0].replace(rand_word, '???', 1))

# tuples_list = []

f = open("train.txt", "w")
for sentence in train_data[:10]:
  rand_word = random.choice(sentence.split())
  masked_sent = re.sub(r"\b%s\b" % re.escape(rand_word), '[MASK]', sentence, count=1)
  f.write(rand_word + ' ' + masked_sent + '\n')
  # tuples_list.append((rand_word, masked_sent))

# f = open("new.txt", "w")
# for sents in tuples_list[0:10]:
#   f.write(str(sents) + '\n')

# print(*tuples_list, sep = "\n") 

# tok_to_ix = load_vocab(tuples_list)

# emb_dim = 5
# num_classes = 3
# learning_rate = 0.01
# model = Net(len(tok_to_ix), emb_dim, num_classes)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss_fn = nn.BCELoss()

# n_epochs = 5
# for epoch in range(n_epochs):
#     model.train()
#     for label, text in tuples_list:
#         x = [tok_to_ix[tok] for tok in text.split()]
#         x_train_tensor = torch.LongTensor(x)
#         # if label == 2:
#         #     y_train_tensor = torch.Tensor([0, 0, 1])
#         # elif label == 1:
#         #     y_train_tensor = torch.Tensor([0, 1, 0])
#         # else:
#         #     y_train_tensor = torch.Tensor([1, 0, 0])
#         y_train_tensor = torch.Tensor([label])
#         pred_y = model(x_train_tensor)
#         loss = loss_fn(pred_y, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     print("\nEpoch:", epoch)
#     print("Training loss:", loss.item())
