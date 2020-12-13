[Enron Email Dataset](https://www.kaggle.com/wcukierski/enron-email-dataset)

#### Install Requirements

```
pip install -r requirements.txt
```

#### Running the model

The repo is already initialized with data to train/test/evaluate the model on.
If you would like to re-write what is in the csv files, edit `preprocess.py`

To run the GPT-2 single sentence autofill demo:

```
python3 gpt2.py
```

To run the GPT-2 test against our email dataset:

```
python3 gpt2-test.py
```

To run the Feed-Forward NN test against our email dataset:

```
python3 nn.py
```
