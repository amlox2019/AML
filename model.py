import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """
    Takes in the entire embedded sentence and returns the hidden states corresponding to each word
    Note: usually the input is a packed sequence to max_sent_len
    """
    def __init__(self, emb_lookup_matrix, hidden_size, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.vocab_size = emb_lookup_matrix.size(0)
        self.emb_dim = emb_lookup_matrix.size(1)
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2
        self.num_layers = 1            
        
        self.device = 'cpu'
        
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb.load_state_dict({'weight': emb_lookup_matrix})
        self.emb.weight.requires_grad = False
        
        self.rnn = nn.GRU(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False, bidirectional=bidirectional)
        
        #self.rnn = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=False, bidirectional=bidirectional)
        
        self.w_s = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, words_in, sent_lens):
        embedded_in = self.emb(words_in)
        
        #batch_size = embedded_in.size(1)
        # We pack the sequence here, since only rnn supports pack_padded
        packed_in = pack_padded_sequence(embedded_in, sent_lens)
        batch_size = embedded_in.size(1)
        
        h_0 = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)
        
        hidden_states, h_n = self.rnn(packed_in, h_0)

        hidden_states, hidden_states_lengths = torch.nn.utils.rnn.pad_packed_sequence(hidden_states) # unpack (back to padded)
        
        if self.bidirectional:
            h_n = h_n[1]
        else:
            h_n = h_n[0]
        last_hidden_state = (self.w_s(h_n)).tanh()
        
        return hidden_states, last_hidden_state
    
    
class AttentionGRUCell(nn.Module):
    """
    Takes in the previous decoded word (embedded via the embedding layer), the last decoded hidden state and the current attended state ; and returns the new hidden state
    It's a modified GRU cell to condition on the attention as well.
    """
    def __init__(self, emb_dim, hidden_size, bidirectional_encoder=True):
        super(AttentionGRUCell, self).__init__() 
        self.input_size = emb_dim
        self.hidden_size = hidden_size
        if bidirectional_encoder:
            self.combined_hidden_size = self.hidden_size * 2
        else:
            self.combined_hidden_size = self.hidden_size
        
        self.w = nn.Linear(self.input_size, self.hidden_size)
        self.w_z = nn.Linear(self.input_size, self.hidden_size)
        self.w_r = nn.Linear(self.input_size, self.hidden_size)
        
        self.u = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_z = nn.Linear(self.hidden_size, self.hidden_size)
        self.u_r = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.c = nn.Linear(self.combined_hidden_size, hidden_size)
        self.c_z = nn.Linear(self.combined_hidden_size, hidden_size)
        self.c_r = nn.Linear(self.combined_hidden_size, hidden_size)
        
        self.u_o = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_o = nn.Linear(self.input_size, self.hidden_size)
        self.c_o = nn.Linear(self.combined_hidden_size, self.hidden_size)
        #self.w_o = nn.Linear(self.hidden_size, self.hidden_size)
        """
        nn.init.normal_(self.w.weight, mean=0, std=0.01)
        nn.init.normal_(self.w_z.weight, mean=0, std=0.01)
        nn.init.normal_(self.w_r.weight, mean=0, std=0.01)
        
        nn.init.orthogonal_(self.u.weight)
        nn.init.orthogonal_(self.u_z.weight)
        nn.init.orthogonal_(self.u_r.weight) 
        
        nn.init.normal_(self.c.weight, mean=0, std=0.01)
        nn.init.normal_(self.c_z.weight, mean=0, std=0.01)
        nn.init.normal_(self.c_r.weight, mean=0, std=0.01)
        
        nn.init.normal_(self.u_o.weight, mean=0, std=0.01)
        nn.init.normal_(self.v_o.weight, mean=0, std=0.01)
        nn.init.normal_(self.c_o.weight, mean=0, std=0.01) 
        
        nn.init.constant_(self.w.bias, 0)
        nn.init.constant_(self.w_z.bias, 0)
        nn.init.constant_(self.w_r.bias, 0)
        nn.init.constant_(self.u.bias, 0)
        nn.init.constant_(self.u_z.bias, 0)
        nn.init.constant_(self.u_r.bias, 0)
        nn.init.constant_(self.c.bias, 0)
        nn.init.constant_(self.c_z.bias, 0)
        nn.init.constant_(self.c_r.bias, 0)   
        nn.init.constant_(self.u_o.bias, 0)
        nn.init.constant_(self.v_o.bias, 0)
        nn.init.constant_(self.c_o.bias, 0)  
        """
        
    def forward(self, in_word, last_hid_state, attended_state):
        
        # * should be element-wise multiplication
        # s_t is the hidden state
        #in_word = in_word.permute(1,0)
        #last_hid_state = last_hid_state.permute(1,0)
        z = (self.w_z(in_word) + self.u_z(last_hid_state) + self.c_z(attended_state)).sigmoid()
        
        r = (self.w_r(in_word) + self.u_r(last_hid_state) + self.c_r(attended_state)).sigmoid()
        
        s_tilda = (self.w(in_word) + self.u(r * last_hid_state) + self.c(attended_state)).tanh()
        
        s = (1 - z) * last_hid_state + z * s_tilda
        
        # TODO: note: this is not quite as in the paper (but close enough); Note: we can do even better than this (e.g. with current-hidden-state-attention)
        t_i = (self.u_o(s) + self.v_o(in_word) + self.c_o(attended_state)).relu()
        return s, t_i
        
        
class Attention(nn.Module):
    """
    Used to attend over the encoded hidden state, using the current decoded hidden state
    """
    def __init__(self, attention_type, hidden_state_size, bidirectional_encoder=True): # Note: only if hidden state size of encoder = hidden state size of decoder.
        super(Attention, self).__init__() 
        self.attention_type = attention_type
        if bidirectional_encoder:
            self.combined_hidden_size = hidden_state_size * 2
        else:
            self.combined_hidden_size = hidden_state_size
        
        if self.attention_type == 'bilinear':
            self.linear = nn.Linear(self.combined_hidden_size, hidden_state_size)
            
        elif self.attention_type == 'paper':
            self.w_a = nn.Linear(hidden_state_size, hidden_state_size)
            self.u_a = nn.Linear(self.combined_hidden_size, hidden_state_size)
            self.tanh = nn.Tanh()
            self.v_a = nn.Linear(hidden_state_size, 1)
            
            """
            nn.init.normal_(self.w_a.weight, mean=0, std=0.001)
            nn.init.normal_(self.u_a.weight, mean=0, std=0.001)
            nn.init.constant_(self.v_a.weight, 0)
            
            nn.init.constant_(self.w_a.bias, 0)
            nn.init.constant_(self.u_a.bias, 0)
            nn.init.constant_(self.v_a.bias, 0)
            """
        
    def attention_weights(self, encoded_states, hidden_state):
        """ 
        I. Dot product of each of the encoded states with the hidden state;
            input: tensor[A,B,C], tensor[B,C]
            output: tensor[A] of dot products 
            
        II. Bilinear attention: (i.e. xMy^t, instead of xy^t)
            input: X=tensor[A,B,C], Y=tensor[B,C]
            output: tensor[A] of elements X[i,:,:]MY
            
        III. Attention as in the paper:
            input: tensor[A,B,C], tensor[B,C]
            output:
            
        """
        max_sent_len = encoded_states.size(0)
        #batch_size = encoded_states.size(1)
        hid_state_size = encoded_states.size(2)
        
        if self.attention_type == 'bilinear':
            tensor1 = encoded_states.permute(1,0,2)
            tensor2 = hidden_state.unsqueeze(2) # add a singular extra dimension

            # batch-wise matrix multiplications; first dim must be batch_size
            tensor3 = self.linear(tensor1)
            attention_weights = torch.bmm(tensor3, tensor2)
            attention_weights = attention_weights.permute(1,0,2)
            
        elif self.attention_type == 'paper':

            h = encoded_states.permute(1,0,2)
            
            # Create max_sent_len copies of the hidden_state, to match the encoded_states
            s = hidden_state

            s = s[:, :, None].permute(0,2,1)
            s = s.expand([s.size(0), h.size(1), s.size(2)])

            att_sum = self.w_a(s) + self.u_a(h)
            
            attention_weights = self.v_a(self.tanh(att_sum))
            attention_weights = attention_weights.permute(1, 0, 2)
        
        else:
            raise NotImplementedError

        return attention_weights
            
        
    def forward(self, encoded_states, hidden_state):  
        max_sent_len = encoded_states.size(0)
        batch_size = encoded_states.size(1)
        hid_state_size = encoded_states.size(2)
        
        # 1) compute the attention weights
        attention_weights = self.attention_weights(encoded_states, hidden_state)

        # 2) softmax of the attention weights
        attention_probs = F.softmax(attention_weights, dim=0)

        # 3) weighed sum of the encoded states by the softmax
        # Note: this is a bit hacky
        # Duplicate attention_probs to multiply it element-wise with the encoded_states
        attention_probs_copies = attention_probs.expand(max_sent_len, batch_size, hid_state_size) # TODO: shouldn't have to expand it!
        attention_probs_copies = attention_probs_copies # pe3rmute the dimensions to match encoded_states
        # elementwise multiplication:
        attended_out = attention_probs_copies * encoded_states
        attended_out = attended_out.sum(dim=0)

        return attended_out, attention_probs


class Decoder(nn.Module):
    """
    Combines the decoder cell with attention over the input hidden states. Returns just one decoded step.
    """
    def __init__(self, emb_lookup_matrix, hidden_size, attention=False, attention_type='paper', cell_type='gru', bidirectional_encoder=True):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = emb_lookup_matrix.size(0)
        self.emb_dim = emb_lookup_matrix.size(1)
        self.attention = attention
        self.attention_type = attention_type
        self.cell_type = cell_type
        
        # The embedding layer is mostly unnecessary
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb.load_state_dict({'weight': emb_lookup_matrix})
        self.emb.weight.requires_grad = False
        
        if self.attention and self.cell_type == 'attention_gru':

            self.cell = AttentionGRUCell(self.emb_dim, self.hidden_size, bidirectional_encoder)
        elif not self.attention and self.cell_type == 'attention_gru':
            self.cell = AttentionGRUCell(self.emb_dim, self.hidden_size, bidirectional_encoder)
            self.w_s = nn.Linear(self.hidden_size, self.hidden_size)
        else:
            self.cell = nn.GRUCell(input_size=self.emb_dim, hidden_size=self.hidden_size)       
            
        if self.attention:
            self.attention_model = Attention(attention_type=self.attention_type, hidden_state_size=self.hidden_size, bidirectional_encoder=bidirectional_encoder)
            
        self.proj_h = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # Note: softmax is inside torch.nn.CrossEntropyLoss (in main.py)       
        
    def forward(self, in_word, encoded_states, last_hid_state):
        in_word = self.emb(in_word)
        # Note: this is a separate case, since in that case, the current attended state comes from the previous hidden state
        if self.attention and self.cell_type == 'attention_gru':
            attended_state, attention_probs = self.attention_model.forward(encoded_states, last_hid_state)
            hidden_state, combined_hidden_state = self.cell(in_word, last_hid_state, attended_state)
                  
        elif self.cell_type == 'gru':
            hidden_state = self.cell(in_word, last_hid_state)
            
            if self.attention:
                attended_state, attention_probs = self.attention_model.forward(encoded_states, hidden_state)
                combined_hidden_state = torch.cat((hidden_state, attended_state), dim=1)
                # Condition the decoding on the attention as well.
                #hidden_state = self.proj_h(combined_hidden_state)
                #print(attention_probs.size(), in_word.size(), encoded_states.size())
                
            else:
                # This is just placeholder
                attention_probs = torch.zeros((encoded_states.size(0), in_word.size(0), 1))
                combined_hidden_state = hidden_state
                
        elif not self.attention and self.cell_type == 'attention_gru':
            enc_last_hid_state = (self.w_s(encoded_states[-1])).tanh() 
            hidden_state, combined_hidden_state = self.cell(in_word, last_hid_state, enc_last_hid_state)
            attention_probs = torch.zeros((encoded_states.size(0), in_word.size(0), 1))
        else:
            raise NotImplementedError
            
        return hidden_state, combined_hidden_state, attention_probs #hidden_state

class EncoderDecoder(nn.Module):
    """
    This is the entire model, with attention. 
    
    The encoder encodes the entire sentence. The decoder just decodes one "word" at a time.  
    
    The beam search should be done outside this class.
    """
    def __init__(self, en_emb_lookup_matrix, fr_emb_lookup_matrix, hidden_size, bidirectional_encoder=True, attention=False, attention_type='paper', decoder_cell_type='gru'):
        super(EncoderDecoder, self).__init__()
        self.bidirectional_encoder = bidirectional_encoder
        self.attention = attention 
        self.attention_type = attention_type
        self.decoder_cell_type = decoder_cell_type
        
        self.encoder = Encoder(en_emb_lookup_matrix, hidden_size, bidirectional=self.bidirectional_encoder)
                
        # Note: if bidirectional, then the encoding hidden size has twice the size
        
        self.decoder_hidden_size = hidden_size
        if self.bidirectional_encoder:
            self.attention_out_dim = hidden_size * 2
        else:
            self.attention_out_dim = hidden_size
        
        if self.attention and decoder_cell_type=='gru':
            self.out_dim = self.decoder_hidden_size + self.attention_out_dim
        else:
            self.out_dim = self.decoder_hidden_size
                   
        self.decoder = Decoder(fr_emb_lookup_matrix, self.decoder_hidden_size, self.attention, self.attention_type, self.decoder_cell_type, self.bidirectional_encoder)

        # Project to vocab_size for prediction
        self.projection = nn.Linear(self.out_dim, self.decoder.vocab_size)      
        
        
    def encode(self, x, sent_lens):
        """this funct is probably obsolete"""
        hidden_states, last_hid_state = self.encoder.forward(x, sent_lens)
        
        return hidden_states, last_hid_state
 
        
    def decode(self, in_word, encoded_states, last_hid_state):
        """this function is probably obsolete"""
        hidden_state, single_out, attention_probs = self.decoder.forward(in_word, encoded_states, last_hid_state)
        # Project the hidden state to vocabulary weights
        single_out = self.projection(single_out)
        
        return single_out, hidden_state, attention_probs

        
        

