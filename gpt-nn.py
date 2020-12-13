
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys 
from halo import Halo
import readline
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Encode a text inputs
# text = "I have a meeting in San Francisco tomorrow"
# predicted_text = text

# -------

spinner = Halo(text='thinking', spinner='dots')

PUNCTUATION = {'.','?','!','~','\n'}

CMD = []

def completer(text, state):
    options = [cmd for cmd in CMD if cmd.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None
readline.parse_and_bind("tab: complete")
readline.set_completer(completer)
def predict(text):
    indexed_tokens = tokenizer.encode(text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model = GPT2LMHeadModel.from_pretrained('gpt2')

    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_indices = torch.topk(predictions[0, -1, :], 4)[1]
    # return tokenizer.decode(indexed_tokens + [predicted_index])
    # return tokenizer.decode([predicted_index])
    # print(tokenizer.decode(predicted_indices))

    return [tokenizer.decode([i.item()]) for i in predicted_indices]


sentence = ''
while sentence == '':
    if sentence == '':
        sentence = input("Start with a few words\n")
    # else:
    #     word = input(sentence)
    #     if word == '\t':
    #         sentence += " TAB"
    #     else:
    #         sentence += f" {word}"
    #
    # spinner.start()
    # suggested_word = predict(sentence)
    # spinner.stop()
    # print(suggested_word)
    text = sentence
    predicted_text = text

    while predicted_text[-1] not in {'.','?','!','~','\n'}:  
        text = predicted_text
        indexed_tokens = tokenizer.encode(text)
        
        # Convert indexed tokens in a PyTorch tensor
        tokens_tensor = torch.tensor([indexed_tokens])
        
        # Load pre-trained model (weights)
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Set the model in evaluation mode to deactivate the DropOut modules
        model.eval()
        
        # If you have a GPU, put everything on cuda
        #tokens_tensor = tokens_tensor.to('cuda')
        #model.to('cuda')
        
        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
        
        # Get the predicted next sub-word
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        print(predicted_text)
    # CMD = [suggested_word]
    # print(suggested_word)
    # next_word = input(sentence + suggested_word[0])
    # if next_word == "":
    #     next_word = suggested_word
    # sentence += next_word








# Print the predicted word
print("predicted word", predicted_text)