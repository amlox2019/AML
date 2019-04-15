"""
This script serves as both qualitative and quantitative evaluation of the trained model(s)
After evaluating the model(s), the plot() function plots the BLEU scores vs the sentence lenths
"""


import os
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import main as mn
from pre import create_data, BatchIterator
from main import evaluate
from model import EncoderDecoder
#from __main__ import *


cwd = os.getcwd() 
#main.device = torch.device('cuda')
lang = 'fr'
train_data_dir = 'datasets/europarl'
# The training data of the trained model
train_dataset = 'w50_europarl-v7.fr-en'

model_name = 'RNNAw50'
mn.model_name = model_name
model_dir = 'saved_models/{}/best_model.pth'.format(model_name)
batch_size = 100
mn.batch_size = batch_size
one_batch = False
mn.max_out_length = 60 # Note: this should be in all cases
use_beam_search = True  # use beam-search for prediction
mn.BEAM_WIDTH = 5
if use_beam_search:
    model_name = model_name + 'beam'
if one_batch:
    model_name += 'one_batch'
mn.model_name = model_name

 

def load_model(model_dir, en_emb_lookup_matrix, target_emb_lookup_matrix):
    save_dict = torch.load(os.path.join(os.path.dirname(cwd), model_dir))
    config = save_dict['config']
    print('  Model config: \n', config)
    model = EncoderDecoder(en_emb_lookup_matrix, target_emb_lookup_matrix, config['h_size'], config['bidirectional'], config['attention'], config['attention_type'], config['decoder_cell_type']).to(device)
    mn.hidden_size = config['h_size']
    model.encoder.device = device
    model.load_state_dict(save_dict['state_dict'])
    return model
    

def translate(dataset, lang):
    train_data, train_source_text, train_target_text = create_data(os.path.join(train_data_dir, train_dataset), lang)
    #main.eval_dataset = 'small_europarl-v7.fr-en'
    
    target_vocab_size = train_target_text.vocab.vectors.size(0)
    mn.target_vocab_size = target_vocab_size
    mn.device = device
    mn.verbose = True

    en_emb_lookup_matrix = train_source_text.vocab.vectors.to(device)
    target_emb_lookup_matrix = train_target_text.vocab.vectors.to(device)

    print('loading model')
    model = load_model(model_dir, en_emb_lookup_matrix, target_emb_lookup_matrix)
    print('model loaded')
    
    d, d_s_v, d_t_v = create_data(dataset, lang)
    d_iter = BatchIterator(d, batch_size, do_train=False, seed=325632)
    
    # Only for debugging and attention weights
    if one_batch == True:
        d_iter.num_batches = 1
    
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    loss, bleu, sent_lens, mean_sent_bleu, sent_bleu, attention_weights = evaluate(model, criterion, d_iter, d_s_v.vocab, d_t_v.vocab, train_source_text.vocab, train_target_text.vocab, print_translations=True, use_beam_search=use_beam_search, corpus_bleu=False)
    print('Test mean loss: {:.3f}; Test bleu score: {:.4f}; Test sent bleu score: {:.4f}'.format(loss, bleu, mean_sent_bleu))
    
    with open(model_name+'_attention_weights', 'wb') as _file:
        pickle.dump(attention_weights, _file)
    
    sent_bleu = [b * 100 for b in sent_bleu]
    sent_lens , sent_bleu = zip(*sorted(zip(sent_lens, sent_bleu)))
    
    print(len(sent_lens), len(sent_bleu))
    sent_lens = np.array(sent_lens)
    sent_bleu = np.array(sent_bleu)
    
    unique_lens = np.unique(sent_lens)
    aggregated_bleu = []
    
    # Go over all unique lengths
    for sent_len in unique_lens:
        # Select all BLEU scores of sentences of sent_len
        ids = np.nonzero(np.array(sent_lens) - sent_len)
        selected_bleu = np.delete(sent_bleu, ids)
        
        mean_bleu = np.mean(selected_bleu)
        aggregated_bleu.append(mean_bleu)
    
    print('Saving data')

    # save this info (so that we don't have to run it every time)
    with open(model_name+'_bleu', 'wb') as _file:
        pickle.dump((unique_lens, aggregated_bleu), _file)
    print('Data saved!')
    
    return unique_lens, aggregated_bleu 
    
    
def plot(model_names, line_styles=['-']):

    for model_name, line_style in zip(model_names, line_styles):
        with open(model_name+'_bleu', 'rb') as _file:
            sent_lens, sent_bleu = pickle.load(_file)
            
        print("--> Plotting figure")

        print(len(sent_lens), len(sent_bleu))
        
        plt.plot(sent_lens, sent_bleu, linestyle=line_style, label=model_name)

    plt.xlabel('Sentence length')
    plt.ylabel('BLEU Score')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim(0, 60)

    print('--> Saving figure')
    plt.savefig(''.join(model_names) + '_sentence_plots.png')
    #plt.savefig('test.png')
    plt.show()
    pass


def main():
    #sent_lens, sent_bleu = translate(args.dataset, args.lang)
    #print('Model evaluated!')
    #plot(model_names=[model_name])
    #models = ['RNNw30', 'RNNw50', 'RNNAw30', 'RNNAw50']
    #line_styles = ['-', '--', '-', '--']
    models = ['RNNAw50', 'RNNAw50beam']
    line_styles = ['-', '-']
    plot(models, line_styles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    eval_data_dir =  os.path.join(os.path.join(os.path.dirname(cwd), 'datasets'),'dev')
    parser.add_argument('--dataset', type = str, default = 'datasets/test/newstest2011')  
    parser.add_argument('--device', type = str, default = 'cpu')
    parser.add_argument('--lang', type = str, default = 'fr')
    args = parser.parse_args()
    global device
    device = args.device

    main()
   

