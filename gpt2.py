import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rich.console import Console

console = Console()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
PUNCTUATION = {'.','?','!','~','\n'}

predicted_text = input("Start with a few words\n")
while predicted_text[-1] not in PUNCTUATION:  
    text = predicted_text
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens])
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
    
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_word = tokenizer.decode([predicted_index])
    console.print(f"{predicted_text}[bold blue]{predicted_word}[/bold blue]")
    predicted_text = f"{predicted_text}{predicted_word}"
