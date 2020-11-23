import pandas as pd
import itertools
import random
import re
import string

train_data = []
df = pd.read_csv("emails.csv", nrows=500)
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))

train_data[:] = [' '.join(x).split('. ') for x in train_data] # i'll fix all this later lol too tired rn
train_data[:] = [[elem for elem in x if elem.strip()] for x in train_data]
train_data[:] = [[elem.strip() for elem in x] for x in train_data]

train_data = list(itertools.chain.from_iterable(train_data))

f = open("train.csv", "w")
f.write('mask,'+'text'+'\n')
for sentence in train_data[:3]:
  split_sent = sentence.split(' ')
  output = [' '.join(split_sent[:2])]
  for word in split_sent[2:]:
    output.append(output[-1] +  ' ' + word)
  for o in output:
    s = o.split(' ')
    f.write(s[-1] + ',' + ' '.join(s[:len(s) - 1]) + '\n')