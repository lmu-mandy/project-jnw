import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


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

updated_data = train_data
train_data[:] = [' '.join(x).split('.') for x in train_data]
train_data[:] = [[elem for elem in x if elem.strip()] for x in train_data]
train_data[:] = [[elem.strip() for elem in x] for x in train_data]

# for str_lists in train_data:
#   str_lists = [elem for elem in str_lists if elem.strip()]
#   print(str_lists)
  # if '' in str_lists:
  #   str_lists.remove("")
  # if ' ' in str_lists:
  #   str_lists.remove(" ")

print(train_data)

# emb_dim = 5
# num_classes = 3
# learning_rate = 0.01
# model = Net(len(tok_to_ix), emb_dim, num_classes)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# loss_fn = nn.BCELoss()

# n_epochs = 5
# for epoch in range(n_epochs):
#     model.train()
#     for text, label in train_data:
#         x = [tok_to_ix[tok] for tok in text.split()]
#         x_train_tensor = torch.LongTensor(x)
#         if label == 2:
#             y_train_tensor = torch.Tensor([0, 0, 1])
#         elif label == 1:
#             y_train_tensor = torch.Tensor([0, 1, 0])
#         else:
#             y_train_tensor = torch.Tensor([1, 0, 0])
#         # y_train_tensor = torch.Tensor([label])
#         pred_y = model(x_train_tensor)
#         loss = loss_fn(pred_y, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#     print("\nEpoch:", epoch)
#     print("Training loss:", loss.item())
