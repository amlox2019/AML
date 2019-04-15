Reproducing: Neural Machine Translation by Jointly Learning to Align and Translate, For Hilary Term 2019, Advanced Machine Learning

Requirements - packages:
-numpy
-pytorch
-torchtext
-spacy
-nltk
-mosestokenizer

Setup the tokenisers:
python -m spacy download en
python -m spacy download fr

Running the code:
python main.py -v

To reproduce the RNN results set:
batch_size = 80
subepoch_size = 2000 (say)
lr = 0.001
max_out_length = 60 (or 40 depending on the train dataset)
bidirectional = False
attention = False
decoder_cell_type = 'attention_gru'
lang = 'fr'

To reproduce the RNN+Attention results set:
batch_size = 80
subepoch_size = 2000 (say)
lr = 0.001
max_out_length = 60 (or 40 depending on the train dataset)
bidirectional = True
attention = True
attention_type = 'paper'
decoder_cell_type = 'attention_gru'
lang = 'fr'

