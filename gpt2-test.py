import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import csv

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

df = pd.read_csv("data/test.csv", quotechar='`')
sentences = []
for i, row in df.iterrows():
    if i != 0:
        text, mask = row['text'], row['mask']
        sentences.append((text, mask))


model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
avg_rank = 0
total = 0
valid = 0
avg_len = 0
with open('gpt2_results.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['accuracy', 'avg_valid_rank', 'rank', 'avg_valid_len', 'len'])
    for sent in sentences:
        text = sent[0]
        indexed_tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        top_k = 10
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        idx = predictions[0, -1, :].topk(top_k).indices.tolist()
        next_words = {tokenizer.decode(
            [j]).strip(): i for i, j in enumerate(idx)}
        next_word_rank = next_words[sent[1]] if sent[1] in next_words else -1
        total += 1
        if next_word_rank >= 0:
            valid += 1
            avg_rank = (avg_rank * (valid-1) + next_word_rank)/valid
            avg_len = (avg_len * (valid-1) + len(text.split()))/valid
        writer.writerow([format(valid/total, '.4f'), format(avg_rank, '.4f'),
                         next_word_rank, format(avg_len, '.4f'), len(text.split())])
