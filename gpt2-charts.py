import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("gpt2_results.csv")
plt.style.use('fivethirtyeight')

# Accuracy over time

plt.plot(data['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Lines Processed')
plt.title('GPT-2: Accuracy Over Time')
plt.show()

# Input word count stacked barchart

valid = data.loc[data['rank'] >= 0]['len'].value_counts().to_dict()
invalid =  data.loc[data['rank'] == -1]['len'].value_counts().to_dict()

valid_counts = []
invalid_counts = []
labels = []
for key in [*valid, *invalid]:
  labels.append(key)
  valid_counts.append(valid.get(key, 0))
  invalid_counts.append(invalid.get(key, 0))

width = 0.35

fig, ax = plt.subplots()

ax.bar(labels, valid_counts, width, label='Valid Prediction')
ax.bar(labels, invalid_counts, width, bottom=valid_counts,  label='Invalid Prediction')  

ax.set_ylabel('Lines Processed')
ax.set_xlabel('Number of Input Words')
ax.set_title('GPT-2: Accuracies of Each Input Word Count')
ax.legend()

plt.show()
