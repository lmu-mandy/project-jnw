import pandas as pd
import itertools
import random
import re
import string

train_data = []
df = pd.read_csv("emails.csv", nrows=10000, usecols=range(2))
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))

train_data[:] = [' '.join(x).split('. ') for x in train_data]
train_data[:] = [[elem for elem in x if elem.strip()] for x in train_data]
train_data[:] = [[elem.strip() for elem in x] for x in train_data]

train_data = list(itertools.chain.from_iterable(train_data))
new_train = []

for idx, value in enumerate(train_data):
  x = train_data[idx]
  if x.replace(' ','').isalpha() and len(x) >= 4:
    new_train.append(x)

train = open("data/train.csv", "w")
train.write('mask,'+'text'+'\n')

test = open("data/test.csv", "w")
test.write('mask,'+'text'+'\n')

val = open("data/val.csv", "w")
val.write('mask,'+'text'+'\n')

for idx, value in enumerate(new_train[:5000]):
  split_sent = new_train[idx].split(' ')
  output = [' '.join(split_sent[:2])]
  for word in split_sent[2:]:
    output.append(output[-1] +  ' ' + word)
  for o in output:
    s = o.split(' ')
    last_word = s[-1].translate(str.maketrans('', '', string.punctuation))
    new_sent = ' '.join(' '.join(s[:len(s) - 1]).split())
    if last_word != '' and not last_word.isdigit() and new_sent != '':
      if idx <= 1670:
        train.write(last_word + ',`' + new_sent + '`\n')
      elif idx <= 3500:
        test.write(last_word + ',`' + new_sent + '`\n')
      else:
        val.write(last_word + ',`' + new_sent + '`\n')