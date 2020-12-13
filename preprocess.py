import pandas as pd
import itertools
import string

# Program to preprocess (remove whitespace, headers, and all lines of text that only contain letters)
# and then split up the preprocessed data into 3 separate csv files (train, test, evaluation)

# Read in 10,000 lines of the email data (start at row 16 to avoid reading in headers)
train_data = []
df = pd.read_csv("emails.csv", nrows=10000, usecols=range(2))
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))         

# remove periods from the data + remove all extra whitespace
train_data[:] = [' '.join(x).split('. ') for x in train_data]
train_data[:] = [[elem for elem in x if elem.strip()] and [elem.strip() for elem in x] for x in train_data]

# convert train_data from list of lists to a list of strings
train_data = list(itertools.chain.from_iterable(train_data))

# if string only contains letters and is longer than 4 words, append to new list
updated_train_data = []
for idx, value in enumerate(train_data):
  x = train_data[idx]
  if x.replace(' ','').isalpha() and len(x) >= 4:
    updated_train_data.append(x)

# create our 3 csv files and initialize them with the 2 column names
train = open("data/train.csv", "w")
train.write('mask,'+'text'+'\n')
test = open("data/test.csv", "w")
test.write('mask,'+'text'+'\n')
val = open("data/val.csv", "w")
val.write('mask,'+'text'+'\n')

# to limit the data we add to the csv files, only read the first 5000 lines
# format the sentence to mask the last word
# start at length 2 and build up until you reach the full length
for idx, value in enumerate(updated_train_data[:5000]):
  split_sent = updated_train_data[idx].split(' ')
  output = [' '.join(split_sent[:2])]
  for word in split_sent[2:]:
    output.append(output[-1] + ' ' + word)
  for o in output:
    s = o.split(' ')
    last_word = s[-1].translate(str.maketrans('', '', string.punctuation))
    new_sent = ' '.join(' '.join(s[:len(s) - 1]).split())
    if last_word != '' and new_sent != '':
      if idx <= 1670:
        train.write(last_word + ',`' + new_sent + '`\n')
      elif idx <= 3500:
        test.write(last_word + ',`' + new_sent + '`\n')
      else:
        val.write(last_word + ',`' + new_sent + '`\n')

# close the files
train.close()
test.close()
val.close()