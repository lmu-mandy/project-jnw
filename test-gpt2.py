"""
This is an experimental file exploring GPT-2 and Enron emails.
"""

import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys 
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

df = pd.read_csv("data/test.csv", quotechar='`')

def predict(text):
    tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([tokens])
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    with torch.no_grad():
        predictions = model(tokens_tensor)[0]

    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    
    # predicted_indices = torch.topk(predictions[0, -1, :], 3)[1]
    # print([tokenizer.decode([i.item()]) for i in predicted_indices])

    return tokenizer.decode([predicted_index])

correct = 0
for i, row in df.iterrows():
    if i != 0:
        text, mask = row['text'], row['mask']
        pred_word = predict(text)
        print()
        print("Given text:", text)
        print("Mask:", mask)
        print("Predicted Word:", pred_word)
        if pred_word.strip(' ') == mask:
          correct += 1
        print(correct, '/', i)