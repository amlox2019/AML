import sys
import os
import argparse
import random

import torch
from torchtext.data import Field, BucketIterator, TabularDataset, Iterator
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import TranslationDataset, WMT14
from mosestokenizer import *

from torchtext.vocab import Vectors

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


cwd = os.getcwd()

small_debug_en = "data_folder"
europarl_training_fr = "datasets/europarl/europarl-v7.fr-en"
test = "datasets/test/newstest2013"
val = "datasets/val/newstest2011"

# Folder with the word embeddings
emb_dir = 'word_embeddings'


def create_data(data, lang):
    source_text = Field(tokenize = MosesTokenizer('en'), init_token='<sos>', eos_token='<eos>', lower=True, pad_token = '<pad>', unk_token = '<unk>')
    target_text = Field(tokenize = MosesTokenizer(lang), init_token='<sos>', eos_token='<eos>', lower=True, pad_token = '<pad>', unk_token = '<unk>')

    train = TranslationDataset(path = data, exts = ('.en', '.' + lang), fields = (source_text, target_text))
    
    # Load the word vectors from the embedding directory
    print('Loading en word vectors')
    en_vectors = Vectors(name='cc.en.300.vec', cache=emb_dir)
    print('Loaded.')
    print('Loading {} word vectors'.format(lang))
    if lang == 'fr':
        target_vectors = Vectors(name='cc.fr.300.vec', cache=emb_dir)
    elif lang == 'de':
        target_vectors = Vectors(name='embed_tweets_de_100D_fasttext', cache=emb_dir)
    else:
        raise NotImplementedError
    print('Loaded.')
     
    # Build vocabulary
    print('Building en vocab')
    source_text.build_vocab(train, max_size=15000, min_freq=1, vectors=en_vectors) 
    print('Building {} vocab'.format(lang))
    target_text.build_vocab(train, max_size=15000, min_freq=1, vectors=target_vectors)   
    #source_text.build_vocab(train, min_freq = 30000, vectors="glove.6B.200d")
    #target_text.build_vocab(train, min_freq = 30000, vectors="glove.6B.200d")

    pad_idx = target_text.vocab.stoi['<pad>']
    print('pad_idx', pad_idx)
    eos_idx = target_text.vocab.stoi['<eos>']
    print('eos_idx', eos_idx)
    
    return train, source_text, target_text


def clean_translation(word_list, eos_id=3):
    """ 
    input: 1-d list (or tensor) of word_ids 
    output: sentence cut until the first <eos>
    """
    #sos_id = 2
    eos_id = 3
    #pad_id = 1
    
    first_eos = 0
    for i, word_id in enumerate(word_list):
        if word_id == eos_id:
            first_eos = i
            break
    if first_eos == 0:
        first_eos = len(word_list)
        
    return word_list[0:first_eos]


def repad_data(data, previous_pad_id=1, to_pad_id=1):
    """ Unpads the data, sorts it by length, pads it back on
        also returns the sentence lengths
    """
    
    sent_list = []
    # go over the batch
    for sent in data.permute(1,0):
        # find non - padded_id indices of the sentence tensor and filter the sentence tensor according to them
        trimmed_sent = sent[(sent - previous_pad_id).nonzero()]
        sent_list.append(trimmed_sent)
    # Sort tensors by length:
    sent_lens = torch.LongTensor([v.size(0) for v in sent_list])
    sent_lens, perm_idx = sent_lens.sort(0, descending=True)
    sent_list = [sent_list[i] for i in perm_idx]
    
    padded_data = pad_sequence(sent_list, batch_first=False, padding_value=to_pad_id)

    return padded_data.squeeze(dim=2), sent_lens, perm_idx
    
    
def add_eos(data, eos_id, previous_pad_id=1, to_pad_id=1):
    # First remove padding
    sent_list = []
    # go over the batch
    for sent in data.permute(1,0):
        # find non - padded_id indices of the sentence tensor and filter the sentence tensor according to them
        trimmed_sent = sent[(sent - previous_pad_id).nonzero()].squeeze(dim=1)
        # Append the <eos> token to the sentence
        eos = torch.LongTensor([eos_id])
        trimmed_sent = torch.cat((trimmed_sent, eos))
        sent_list.append(trimmed_sent)
        
    # Add padding back on
    padded_data = pad_sequence(sent_list, batch_first=False, padding_value=to_pad_id)    
    return padded_data
    

class BatchIterator():
    def __init__(self, dataset, batch_size, do_train, seed=1):
        super(BatchIterator, self).__init__()
        self.batch_size = batch_size
        self.do_train = do_train
        
        random.seed(seed)
        
        # We need different iterators for train and eval
        if self.do_train:     
            iterator = BucketIterator(dataset=dataset, batch_size=self.batch_size, train=True, sort_key=lambda x: torchtext.data.interleave_keys(len(x.src), len(x.trg)))
 
        else:
            iterator = Iterator(dataset=dataset, batch_size=self.batch_size, sort=False, sort_within_batch=False, repeat=False)      
        
        self.iterator = iterator     
        self.num_batches = len(iterator)
        
        self.iter = iter(self.iterator)
    

    def batches(self, eos_id):
        batch = next(self.iter)
        input_en = batch.src
        output_fr = batch.trg
        
        # extract the sentence lengths, and rearrange the input data according to sentence lengths (rearrange output accordingly)
        input_en, sent_lens, perm_idx = repad_data(input_en)
        output_fr = output_fr[1:,perm_idx]

        # Add the <eos> token to the target sentences (not needed!)
        #output_fr = add_eos(output_fr, eos_id, previous_pad_id=1, to_pad_id=1)
        
        return input_en, sent_lens, output_fr



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='data_folder')
    parser.add_argument('--language', type=str, default='fr')
    parser.add_argument('--batch_size', type=int, default=1220)

    args = parser.parse_args()


    train, src_vocab, trg_vocab = create_data(args.dataset, args.language)

    batch_iterator = BatchIterator(train, args.batch_size)
    # Get next batch
    input, output = batch_iterator.batches(train, args.batch_size)

