import pandas as pd
import pip._internal.resolution.resolvelib.resolver
import string

# Read in the email data (start at row 16 to avoid reading in headers)
train_data = []
df = pd.read_csv("emails.csv", nrows=10000, usecols=range(2))
for index, row in df.iterrows():
    train_data.append((row['message'].splitlines()[16:]))         

# remove periods from the data + remove all extra whitespace
train_data[:] = [' '.join(x).split('. ') for x in train_data]
train_data[:] = [[elem for elem in x if elem.strip()] and [elem.strip() for elem in x] for x in train_data]

# convert train_data from list of lists to a list of strings
train_data = list(itertools.chain.from_iterable(train_data))

# create a new list that will hold all of the data that we want (text only)
updated_train_data = []

for idx, value in enumerate(train_data):
  x = train_data[idx]
  # if string only contains letters and is longer than 4 words, append to new list
  if x.replace(' ','').isalpha() and len(x) >= 4:
    updated_train_data.append(x)

# create our 3 csv files and initialize them with the 2 column names
train = open("data/train.csv", "w")
train.write('mask,'+'text'+'\n')
test = open("data/test.csv", "w")
test.write('mask,'+'text'+'\n')
val = open("data/val.csv", "w")
val.write('mask,'+'text'+'\n')

for idx, value in enumerate(updated_train_data[:5000]):
  split_sent = updated_train_data[idx].split(' ')
  # mask the last word in the sentence fragment
  output = [' '.join(split_sent[:2])]
  for word in split_sent[2:]:
    output.append(output[-1] + ' ' + word)
  for o in output:
    s = o.split(' ')
    # take the last word in the sentence fragment and mask it
    last_word = s[-1].translate(str.maketrans('', '', string.punctuation))
    # takes the sentence fragment without the last word
    new_sent = ' '.join(' '.join(s[:len(s) - 1]).split())
    # split the sentences into 3 different csv files
    # only add them if both the last word and the new sentence fragment aren't empty strings
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