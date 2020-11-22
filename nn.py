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
df = pd.read_csv("data/labeled_data.csv")
for index, row in df.iterrows():
    train_data.append((row['tweet'], row['class']))
tok_to_ix = load_vocab(train_data)

emb_dim = 5
num_classes = 3
learning_rate = 0.01
model = Net(len(tok_to_ix), emb_dim, num_classes)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()

n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    for text, label in train_data:
        x = [tok_to_ix[tok] for tok in text.split()]
        x_train_tensor = torch.LongTensor(x)
        if label == 2:
            y_train_tensor = torch.Tensor([0, 0, 1])
        elif label == 1:
            y_train_tensor = torch.Tensor([0, 1, 0])
        else:
            y_train_tensor = torch.Tensor([1, 0, 0])
        # y_train_tensor = torch.Tensor([label])
        pred_y = model(x_train_tensor)
        loss = loss_fn(pred_y, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("\nEpoch:", epoch)
    print("Training loss:", loss.item())

# y_test = 'sad face'
# with torch.no_grad():
#     model.eval()
#     x = [tok_to_ix[tok] for tok in y_test.split() if tok in tok_to_ix]
#     x_test = torch.LongTensor(x)
#     pred_y_test = model(x_test)

#     print('Test sentence:', y_test)
#     print('Predicted label:', pred_y_test.data.numpy())
