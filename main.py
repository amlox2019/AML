import os

import argparse
import time
from collections import namedtuple
from copy import deepcopy

from random import randint
from math import floor
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EncoderDecoder
from partial_sentence import PartialSentence
from metric_store import Metric_Store
from pre import create_data, BatchIterator, clean_translation
from metrics import getsentence_bleuscore, getcorpus_bleuscore


# The save directory is one level down from the code directory
cwd = os.getcwd()
save_dir = os.path.join(os.path.dirname(cwd), 'saved_models')
model_name = 'some_model'
save_dir = os.path.join(save_dir, model_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_dir = 'datasets'
train_data_dir = data_dir
dev_data_dir = os.path.join(data_dir, 'dev')
train_dataset = 'europarl/w50_europarl-v7.fr-en' #w30_europarl-v7.fr-en'
eval_dataset = 'newstest2012_2013'

resume = False  # resume training from checkpoint or not.

BEAM_WIDTH = 5
batch_size = 80
subepoch_size = 2#2000 # cut the epoch into several sub-epochs of subepoch_size
lr = 0.001
max_out_length = 40 # max length of decoded sentence

hidden_size = 256 #256  # size of hidden states of encoder; the decoder's hidden size is twice that if the encoder is bidirectional           
bidirectional = False  # bidirectional (rnn) encoder or not
attention = False  # use attention or vanilla rnn
attention_type = 'paper' # 'paper' , 'bilinear'
decoder_cell_type = 'attention_gru' #'attention_gru' # 'attention_gru', 'gru'

print('Process id: ', os.getpid())
print('Model: ', model_name)
print(' Train dataset: {} \n Eval dataset: {}'.format(train_dataset, eval_dataset))
print('  b_size: {} \n  beam: {} \n  l_rate: {} \n  h_size: {} \n  bidirectional: {} \n  attention: {} {} \n  cell: {}'.format(batch_size, BEAM_WIDTH, lr, hidden_size, bidirectional, attention, attention_type, decoder_cell_type))

num_epochs = 20  # number of passes over the whole data
assert num_epochs is not None

lang = 'fr'  # The target text language

eos_id = 3 # The <eos> word id in the target vocab



metric_store = Metric_Store()

start_time = time.time()

def print_runtime_metric(action):
    global start_time
    now = time.time()
    delta = now - start_time
    print('{}: {:.3f}s'.format(action, delta))
    start_time = now

def get_beam_tensors(batch):
    in_words = torch.stack([ps.in_word for ps in batch]).view(-1)
    last_hidden_states = torch.stack([lhs for ps in batch for lhs in ps.last_hidden_state])
    return in_words.to(device), last_hidden_states.to(device)


def beam_predict(model, word_ids, sent_lens):
    encoded_states, last_hidden_state = model.encode(word_ids, sent_lens)
    num_steps = 1
    pred_batch_size = word_ids.size(1)
    if pred_batch_size != batch_size:
        print("WARNING: irregular batch size")
    
    out = []
    predictions = []
    # if we initialise with the beam width then we'll just get the same answer 5 times
    beam = []
    for i in range(pred_batch_size):
        beam.append(PartialSentence(
            torch.tensor([0]),
            [last_hidden_state[i]],
            [torch.tensor([]) for _ in range(BEAM_WIDTH)],
            [torch.tensor([], dtype=torch.long) for _ in range(BEAM_WIDTH)],
            torch.tensor([0.]*BEAM_WIDTH),
            [torch.tensor([]) for _ in range(BEAM_WIDTH)]
        ))

    in_words, last_hidden_states = get_beam_tensors(beam)

    for _ in range(max_out_length):
        new_beam = []

        # Make sure the encoded_states are of the same batch size as the rest.
        if _ == 1:
            encoded_states = encoded_states.repeat(1, BEAM_WIDTH, 1)
        assert encoded_states.size(1) == in_words.size(0) and encoded_states.size(1) == last_hidden_states.size(0)
        
        single_out, hidden_state, attention_weights = model.decode(in_words, encoded_states, last_hidden_states)

        # Here we predict next word in the decoding via softmax+argmax
        pred_probs = F.softmax(single_out, dim=1)
        batch_values, batch_predictions = pred_probs.topk(BEAM_WIDTH, dim=1)

        single_out = single_out.view(pred_batch_size,-1,target_vocab_size)
        batch_values = batch_values.view(pred_batch_size,-1)
        batch_predictions = batch_predictions.view(pred_batch_size,-1)
        hidden_state = hidden_state.view(pred_batch_size,-1,hidden_size)
        attention_weights = attention_weights.transpose(0,1)
        attention_weights = attention_weights.view(pred_batch_size, -1, attention_weights.size()[1])

        for ps, hs, predictions, value, out, a_w in zip(beam, hidden_state, batch_predictions, batch_values, single_out, attention_weights):
            parent_indices = [0] if value.size(0) == BEAM_WIDTH else list(range(BEAM_WIDTH))
            parent_indices = torch.tensor([[i]*BEAM_WIDTH for i in parent_indices]).view(-1) #.to(device)
                
            parent_scores = torch.tensor([ps.score[i] for i in parent_indices]).to(device)
            n = [len(o) + 1 for o in ps.out]
            score = parent_scores + value.log()

            _, indices = score.topk(BEAM_WIDTH)
            score = [score[i] for i in indices]
            parent_indices = [parent_indices[i] for i in indices]

            new_in_word = torch.stack([predictions[i] for i in indices])
            new_hidden_state = [hs[i] for i in parent_indices]

            new_out = [[*ps.out[i], out[i]] for i in parent_indices]
            
            parent_predictions = [ps.predictions[i] for i in parent_indices]
            predictions = [predictions[i] for i in indices]

            new_predictions = [[*prev_predictions,new_prediction] for prev_predictions,new_prediction in zip(parent_predictions, predictions)]
            
            new_attention_weights = [[*ps.attention_weights[i], a_w[i]] for i in parent_indices]
            
            new_beam.append(PartialSentence(
                in_word = new_in_word,
                last_hidden_state = new_hidden_state,
                out = new_out,
                predictions = new_predictions,
                score = score,
                attention_weights = new_attention_weights
            ))

        beam = new_beam
        in_words, last_hidden_states = get_beam_tensors(beam)
    
    out = [torch.stack(ps.out[0]) for ps in beam]
    predictions = [torch.stack(ps.predictions[0]) for ps in beam]
    attention_weights = [torch.stack(ps.attention_weights[0]) for ps in beam]
    attention_weights = torch.stack(attention_weights, dim=0)
    #print(word_ids.size(), out[0].size())
    out = torch.stack(out, dim=0).permute(1, 0, 2)
    print(attention_weights.size(), len(attention_weights))
    return out, predictions, attention_weights


# Overriding beam search for far greater speed
def predict(model, word_ids, sent_lens):
    encoded_states, last_hidden_state = model.encode(word_ids, sent_lens)
    
    pred_batch_size = word_ids.size(1)
    if pred_batch_size != batch_size:
        print("WARNING: irregular batch size")

    num_steps = 1
    
    out = []
    predictions = []
    attention_weights = []
    #in_word = torch.zeros(batch_size)
    
    # This should be the <start> decoder token (first input to decoder)
    sos_id = 2
    in_word = [2 for i in range(pred_batch_size)]
    in_word = torch.tensor(in_word).to(device)
   
    
    while num_steps <= max_out_length:
        #print('Step', num_steps)
        single_out, hidden_state, attention_weight = model.decode(in_word, encoded_states, last_hidden_state)
        out.append(single_out)
        last_hidden_state = hidden_state

        # Here we predict next word in the decoding via softmax+argmax
        pred_probs = F.softmax(single_out, dim=1)
        values, prediction = pred_probs.max(dim=1) # this is argmax
        predictions.append(prediction)
        
        if num_steps == 1:
            print(attention_weight.size())
        attention_weights.append(attention_weight)
        # the predicted word is fed back to the decoder
        in_word = prediction
        num_steps += 1
    out = torch.stack(out, dim=0)
    predictions = torch.stack(predictions, dim=0).permute(1,0)
    attention_weights = torch.stack(attention_weights, dim=0).permute(2,0,1,3).squeeze(dim=3)
    print(attention_weights.size())
    return out, predictions, attention_weights
    

def lookup_words(w, vocab):
    #print('w', w)
    w = w.cpu().numpy()
    w = [vocab.itos[i] for i in w]
    return [str(t) for t in w]


class BatchReindexer():
    def __init__(self, eval_vocab, train_vocab):
        super(BatchReindexer, self).__init__()
        self.eval_vocab = eval_vocab
        self.train_vocab = train_vocab
        
    def reindex(self, eval_word_id):
        """ Reindex a word_id from the eval_vocab to train_vocab """
        word = self.eval_vocab.itos[eval_word_id]
        train_word_id = self.train_vocab.stoi[word]
        return train_word_id        
    
    def reindex_batch(self, batch):
        reindex = lambda x: self.reindex(x)
        return batch.apply_(reindex)
        

def trim_data(out, target_data):
    """ trim the data to match in dimensions """

    # This is the maximal sentence lenght over both target and output        
    min_len = min(out.size(0), target_data.size(0))

    out = torch.narrow(out, 0, 0, min_len)
    target_data = torch.narrow(target_data, 0, 0, min_len)

    return out, target_data
    
    
def train(model, criterion, optimiser, data_iterator, source_vocab, target_vocab, subepoch, num_subepochs):
    losses = []
    
    avg_batch_time = 0
    batch_history = []
    for i in range(subepoch_size):
        batch_start = time.time()
        model.train() # enable dropout and other training-specific layers
        model.zero_grad() # do this, since PyTorch accumulates gradients by default
        t = time.time()
        input_data, sent_lens, target_data = data_iterator.batches(eos_id)
        it_time = time.time()-t

        input_data, target_data = input_data.to(device), target_data.to(device)

        t = time.time()
        out, predictions, attention_weights = predict(model, input_data, sent_lens)

        # Trim the data to allow for comparison
        out, target_data = trim_data(out, target_data)         

        # We are flattening both the target and the output, as in the criterion we need to average both over epoch, and over decoding timesteps (output positions)
        target_data = target_data.flatten()
        out = out.flatten(start_dim=0, end_dim=1)
        
        loss = criterion(out, target_data)
        losses.append(loss.item())
        pred_time = time.time()-t

        batch_time = time.time() - batch_start
        batch_history.append(batch_time)
            #avg_batch_time = (avg_batch_time * i + batch_time) / (i + 1)
        avg_batch_time = np.mean(batch_history[-15:])
        remaining_batches = (subepoch_size - i - 1) + (num_subepochs - subepoch) * subepoch_size
        epoch_time_remaining = remaining_batches * avg_batch_time
                   
        if verbose and (i+1) % 10 == 0:
            print('  Batch: {}/{} ; Loss: {:.2f}'.format(i+1, subepoch_size, loss.item()))
            print('Est time per ep: {:.2f}s'.format(epoch_time_remaining))

        t = time.time()
        loss.backward()  # backprop the gradient
        optimiser.step()  # calculate parameter updates
        back_time = time.time()-t
        
        if verbose and (i+1) % 10 == 0:
            print('Times: iter {} pred {} backward {}'.format(round(it_time,2), round(pred_time,2), round(back_time,2)))
    mean_loss = np.array(losses).mean().item()
        
    return mean_loss
    

def evaluate(model, criterion, data_iterator, source_eval_vocab, target_eval_vocab, source_train_vocab, target_train_vocab, print_translations=False, use_beam_search=False, corpus_bleu=False):

    src_batch_reindexer = BatchReindexer(source_eval_vocab, source_train_vocab)
    trg_batch_reindexer = BatchReindexer(target_eval_vocab, target_train_vocab)
    
    sent_len_list = []
    
    avg_batch_time = 0
    model.eval()  # disables dropout, etc.
    with torch.no_grad():  # disables backprop
        num_batches = data_iterator.num_batches
        losses = []
        bleu_score = []
        sent_bleu = []
        
        with open(model_name + '.txt', 'w+') as filetowrite:     
        
            for i in range(num_batches):
                batch_start = time.time()
                input_data, sent_lens, target_data = data_iterator.batches(eos_id)
                
                sent_len_list += sent_lens.tolist()
                         
                # reindex the eval data w.r.t. the train vocabulary
                # This is needed since the eval data has a separate vocab
                t1 = time.time()
                input_data = src_batch_reindexer.reindex_batch(input_data)
                target_data = trg_batch_reindexer.reindex_batch(target_data)
                t11 = t1-time.time()
                    
                input_data, target_data = input_data.to(device), target_data.to(device)
                t = target_data.transpose(0,1)
                
                t2 = time.time()
                if use_beam_search:
                    out, predictions, attention_weights = beam_predict(model, input_data, sent_lens)
                else:
                    out, predictions, attention_weights = predict(model, input_data, sent_lens)
                t22 = t2 - time.time()
                
                t3 = time.time()

                # Trim the data to allow for comparison
                out, target_data = trim_data(out, target_data) 
                t33 = t3 - time.time() 
                   
                # We are flattening both the target and the output, as in the criterion we need to average both over epoch, and over decoding timesteps (output positions)
                target_data = target_data.flatten()
                out = out.flatten(start_dim=0, end_dim=1)
                
                t4 = time.time()
                hypothesis = [lookup_words(clean_translation(words), target_train_vocab) for words in predictions]
                hyp = [" ".join(x) for x in hypothesis]

                source = [lookup_words(words, source_train_vocab) for words in input_data.transpose(0,1)]                       
                src = [" ".join(x) for x in source]
                
                references = [[lookup_words(clean_translation(words), target_train_vocab)] for words in t]

                ref = [" ".join(x[0]) for x in references]  
                
                if corpus_bleu:
                    bleu = getcorpus_bleuscore(references, hypothesis)
                else:
                    bleu = 0
                bleu_score.append(bleu)
                t44 = t4 - time.time()
                #print(t11,t22,t33,t44)
                # to print sentence-wise translations
                examples = zip(input_data.transpose(0,1), t, predictions)
                for sourc, target, x in examples:
                    t5 = time.time()
                    translation = lookup_words(clean_translation(x), target_train_vocab)
                    references = lookup_words(clean_translation(target), target_train_vocab)
                    source = lookup_words(clean_translation(sourc), source_train_vocab)        
                    t55 = t5 - time.time()
                    # This is to assure that no division by zero happens in sentence BLEU scoring.
                    if len(translation) == 0:
                        translation = ['some_token']    
                    #print('Source: "{}"\nReferences:\n{}\nTranslation: "{}"\n'.format(source, references, translation))
                    t6 = time.time()
                    
                    filetowrite.write('Source: "{}"\nReferences:\n{}\nTranslation: "{}"\n'.format(' '.join(source), ' '.join(references), ' '.join(translation)))
                    t66 = t6 - time.time()
                    t7 = time.time()
                    bleuscore = getsentence_bleuscore(references, translation)
                    if print_translations:
                        print(' Reference: {} \n Hypothesis: {} '.format(' '.join(references), ' '.join(translation)))

                        print('BLEU score: {} \n'.format(bleuscore))
                    sent_bleu.append(bleuscore)
                    t77 = t7 - time.time()
                    #print(t55, t66, t77)
               # Fixed-length output; first average over the output sentence to obtain the loss J, then average over the batch         
                loss = criterion(out, target_data)
                losses.append(loss.item())
                
                if verbose:
                    batch_time = time.time() - batch_start
                    avg_batch_time = (avg_batch_time * i + batch_time) / (i + 1)
                    remaining_batches = num_batches - i - 1
                    epoch_time_remaining = remaining_batches * avg_batch_time
                    print('  Batch: {}/{} ; Loss: {:.3f}; Bleu: {:.4f}; sent BLEU: {:.4f}'.format(i+1, num_batches, loss.item(), bleu, np.mean(sent_bleu[-len(predictions):])))
                    print('Estimated time remaining for epoch: {:.2f}s'.format(epoch_time_remaining))

            
        filetowrite.close()
           
        mean_loss = torch.tensor(losses).mean().item()
        mean_bleu = np.mean(bleu_score)
        mean_sent_bleu = np.mean(sent_bleu)

        attention_plot = {
            'source': src,
            'translation': hyp,
            'weights': attention_weights
        }

    return mean_loss, mean_bleu, sent_len_list, mean_sent_bleu, sent_bleu, attention_plot


def main():
    torch.manual_seed(10)  # fix seed for reproducibility
    torch.cuda.manual_seed(10)

    train_data, train_source_text, train_target_text = create_data(os.path.join(train_data_dir, train_dataset), lang)
    #dev_data, dev_source_text, dev_target_text = create_data(os.path.join(eval_data_dir, 'newstest2012_2013'), lang)

    eval_data, eval_source_text, eval_target_text = create_data(os.path.join(dev_data_dir, eval_dataset), lang)
    
    en_emb_lookup_matrix = train_source_text.vocab.vectors.to(device)
    target_emb_lookup_matrix = train_target_text.vocab.vectors.to(device)

    global en_vocab_size
    global target_vocab_size

    en_vocab_size = train_source_text.vocab.vectors.size(0)
    target_vocab_size = train_target_text.vocab.vectors.size(0)

    if verbose:
        print('English vocab size: ', en_vocab_size)
        print(lang, 'vocab size: ', target_vocab_size)
        print_runtime_metric('Vocabs loaded')

    model = EncoderDecoder(en_emb_lookup_matrix, target_emb_lookup_matrix, hidden_size, bidirectional, attention, attention_type, decoder_cell_type).to(device)

    model.encoder.device = device

    criterion = nn.CrossEntropyLoss(ignore_index=1) # ignore_index=1 comes from the target_data generation from the data iterator
    
    #optimiser = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0) # This is the exact optimiser in the paper; rho=0.95
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = 10e+10  # dummy variable
    best_bleu = 0 
    epoch = 1  # initial epoch id
    
    if resume:
        print('\n ---------> Resuming training <----------')
        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        subepoch, num_subepochs = checkpoint['subepoch_num']
        model.load_state_dict(checkpoint['state_dict'])
        best_loss = checkpoint['best_loss']
        optimiser.load_state_dict(checkpoint['optimiser'])
        is_best = checkpoint['is_best']
        metric_store.load(os.path.join(save_dir, 'checkpoint_metrics.pickle'))
        
        if subepoch == num_subepochs:
            epoch += 1
            subepoch = 1
        else:
            subepoch += 1
    
    if verbose:
        print_runtime_metric('Model initialised')

    while epoch <= num_epochs:
        is_best = False # best loss or not
        
        # Initialise the iterators
        train_iter = BatchIterator(train_data, batch_size, do_train=True, seed=epoch**2)
        
        num_subepochs = train_iter.num_batches // subepoch_size
        
        # train sub-epochs from start_batch
        # This allows subepoch training resumption
        if not resume:
            subepoch = 1
        while subepoch <= num_subepochs:
            
            if verbose:
                print(' Running code on: ', device)

                print('------> Training epoch {}, sub-epoch {}/{} <------'.format(epoch, subepoch, num_subepochs))
           
            mean_train_loss = train(model, criterion, optimiser, train_iter, train_source_text, train_target_text, subepoch, num_subepochs)

            if verbose: 
                print_runtime_metric('Training sub-epoch complete')
                print('------> Evaluating sub-epoch {} <------'.format(subepoch))
                
            eval_iter = BatchIterator(eval_data, batch_size, do_train=False, seed=325632)
            
            mean_eval_loss, mean_eval_bleu, _, mean_eval_sent_bleu, _, _ = evaluate(model, criterion, eval_iter, eval_source_text.vocab, eval_target_text.vocab, train_source_text.vocab, train_target_text.vocab) # here should be the eval data
            
            if verbose:
                print_runtime_metric('Evaluating sub-epoch complete')

            if mean_eval_loss < best_loss:
                best_loss = mean_eval_loss
                is_best=True
             
            if mean_eval_bleu > best_bleu:
                best_bleu = mean_eval_bleu
                is_best=True
            
            config_dict = {
                'train_dataset': train_dataset,
                'b_size': batch_size, 
                'h_size': hidden_size, 
                'bidirectional': bidirectional,
                'attention': attention, 
                'attention_type': attention_type, 
                'decoder_cell_type': decoder_cell_type
                }
                
            # Save the model and the optimiser state for resumption (after each epoch)
            checkpoint = {
                'epoch': epoch,
                'subepoch_num': (subepoch, num_subepochs),
                'state_dict': model.state_dict(),
                'config': config_dict,
                'best_loss': best_loss,
                'best_BLEU': best_bleu,
                'optimiser' : optimiser.state_dict(),
                'is_best' : is_best
            }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
            metric_store.log(mean_train_loss, mean_eval_loss)
            metric_store.save(os.path.join(save_dir, 'checkpoint_metrics.pickle'))

            if verbose:
                print('Checkpoint.')
            
            # Save the best model so far
            if is_best:
                save_dict = {
                    'state_dict': model.state_dict(),
                    'config': config_dict,
                    'epoch': epoch
                    }
                torch.save(save_dict, os.path.join(save_dir, 'best_model.pth'))
                metric_store.save(os.path.join(save_dir, 'best_model_metrics.pickle'))

            if verbose:    
                if is_best:
                    print('Best model saved!')
                print('Ep {} Sub-ep {}/{} Tr loss {} Eval loss {} Eval BLEU {} Eval sent BLEU {}'.format(epoch, subepoch, num_subepochs, round(mean_train_loss, 3), round(mean_eval_loss, 3), round(mean_eval_bleu, 4), round(mean_eval_sent_bleu,4)))
                
            subepoch += 1
        epoch += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    global device
    global verbose

    device = args.device
    verbose = args.verbose
    main()
    

