import pandas as pd
import itertools
import random
import re
import string

train_data = []
df = pd.read_csv("emails.csv", nrows=10000, usecols=range(2))
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))

train_data[:] = [' '.join(x).split('. ') for x in train_data] # i'll fix all this later lol too tired rn
train_data[:] = [[elem for elem in x if elem.strip()] for x in train_data]
train_data[:] = [[elem.strip() for elem in x] for x in train_data]

train_data = list(itertools.chain.from_iterable(train_data))

for sents in train_data:
  if '---' in sents:
    # print(sents)
    train_data.remove(sents)

f = open("train.csv", "w")
f.write('mask,'+'text'+'\n')
for sentence in train_data[:10000]:
  split_sent = sentence.split(' ')
  output = [' '.join(split_sent[:2])]
  for word in split_sent[2:]:
    output.append(output[-1] +  ' ' + word)
  for o in output:
    s = o.split(' ')
    last_word = s[-1].translate(str.maketrans('', '', string.punctuation))
    new_sent = ' '.join(' '.join(s[:len(s) - 1]).split())
    if last_word != '' and not last_word.isdigit() and new_sent != '':
      f.write(last_word + ',' + new_sent + '\n')