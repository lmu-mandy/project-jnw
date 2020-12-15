#### Building a model (FFNN) and testing a pre trained model (GPT-2) on next word prediction.

#### Testing with the [Enron Email Dataset](https://www.kaggle.com/wcukierski/enron-email-dataset)

---

#### Install Requirements

```
pip install -r requirements.txt
```

#### Processing the data

The `data` directory includes post-proccessed data as `train`, `test` and `val` csv files. If you would like to test/run on new data or edit the amount of lines being read, edit and run `preprocess.py`

#### Running the models

To run the GPT-2 single sentence autofill demo:

```
python3 gpt2.py
```

To run the GPT-2 test against our email dataset:

```
python3 gpt2-test.py
```

> Results are written to `gpt2_results.csv`

To run and view our GPT-2 Charts:

```
python3 gpt2-charts.py
```

To run the Feed-Forward NN test against our email dataset:

```
python3 nn.py
```
