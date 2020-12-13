import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("gpt2_results.csv")

plt.plot(data['accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Tests')
plt.show()
