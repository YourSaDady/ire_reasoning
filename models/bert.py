import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random as rand
import math
import itertools
import transformers
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
from pathlib import Path
os.chdir('/home/yichuan/HKU/EBM')
print(f'The current working directory: {os.getcwd()}')

'''
A simplified version of BERT and its relative classes from scratch.

Train like LlaDA (Masked Diffusion)

Simplified parts (assumptions):
    - no is_next_label flag and random get_item
'''

class PositionalEmbedding(nn.Module):
    '''
    Using sine-cosine position embedding
    
    pos_emb: Size(max_len, d_model), where d_model is the output dimension
    '''
    def __init__(self, d_model, max_len=128):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):   
            # for each dimension of the each position
            for i in range(0, d_model, 2):   
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)   
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe
    
class BERTEmbedding(nn.Module):
    '''
    Consisting with:
        - TokenEmbedding
        - PositionalEmbedding
        - SegmentEmbedding
        
    Take sum
    '''
    def __init__(self, vocab_size=7, embed_size=128, seq_len=256, drop_out=0.1):
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len)
        self.dropout = nn.Dropout(p=drop_out)
       
    def forward(self, sequence, segment_label):
        # print(f'Inside BERTEmbedding.forward(), input sequence.shape: {sequence.shape}')
        x = self.token(sequence) + self.position(sequence) # + self.segment(segment_label)
        return self.dropout(x) 
    

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, stage, max_len=256):
        '''
        Init params:
            - data_pair: a list of (x, y) pairs, where x and y are strings, and words are separated by blank spaces
            - stage: pretrain / sft specify the masking area when calling __getitem__()
        '''
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.corpus_lines = len(data_pair)
        self.lines = data_pair
        self.stage = stage
        self.is_pos = True
        
    def __len__(self):
        return self.corpus_lines
    
    def __getitem__(self, item): #"0", "1", MASK only, no segment label
        '''
        Select a positive sample pair from the data_pair, and preprocess.  
        
        params:
            - item: line index
            - t: a random masking rate from a specific schedule
            - is_pos: positive means x and y from the same line pair; otherwise negative
            
        output_dict:
            - bert_input: masked context tensor of Size(max_len)
            - bert_label: padded label tensor of Size(max_len)
            - is_positive: bool, indicating whether is positive or not 
        '''
        t2_all_zero = True
        while t2_all_zero: #randomly mask at least one position in t2
            # uniformly sample unmasking rate t
            t = random.random()
            if t == 0:
                t = 0.05
            # get pos / neg sentence pair
            t1, t2 = self.get_sent(item, self.is_pos)
            # print(f'Inside __getitem__(), original t1: \n{t1}, \nt2: \n{t2}')
            # randomly mask t1 and t2 with a uniformly sampled masking rate t
            if self.stage == 'pretrain':
                (t1, t1_label), (t2, t2_label) = self.mask(t1, t), self.mask(t2, t) #tokenized
                t2_all_zero = all(x == 0 for x in t2_label)
            # mask t2 with random t while keep t1 unmasked
            elif self.stage == 'sft':
                (t1, t1_label), (t2, t2_label) = self.mask(t1, 0), self.mask(t2, t) #t
                t2_all_zero = all(x == 0 for x in t2_label)
            # keep t1 unmasked and t2 fully masked as the initial sequence 
            elif self.stage == 'inference':
                (t1, t1_label), (t2, t2_label) = self.mask(t1, 0), self.mask(t2, 1)
                t2_all_zero = False
            else:
                raise NotImplementedError
        
        # print(f'After added special tokens, t1({len(t1)}): \n{t1}, \nt1_label: \n{t1_label}, \nt2({len(t2)}): {t2}, \nt2_label: {t2_label}')
        # concatenate t1 and t2 and add padding to max_len
        # print(f'\nself.max_len: {self.max_len}')
        bert_input = (t1 + t2)[:self.max_len]
        bert_label = (t1_label + t2_label)[:self.max_len]
        padding = [self.tokenizer.vocab['[MASK]'] for _ in range(self.max_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding)
        
        
        output = {
            'bert_input': bert_input,
            'bert_label': bert_label,
        }
        # print(f'final:\nbert_input: \n{bert_input}, \nbert_label: \n{bert_label},\nsegment_label: \n{segment_label}')
        output = {k: torch.tensor(v) for k, v in output.items()}
        output['is_positive'] = self.is_pos
        
        return output

    def __getitem__complete(self, item): #includes SEP, CLS, PAD tokens
        '''
        Select a positive sample pair from the data_pair, and preprocess.  
        
        params:
            - item: line index
            - t: a random masking rate from a specific schedule
            - is_pos: positive means x and y from the same line pair; otherwise negative
            
        output_dict:
            - bert_input: masked context tensor of Size(max_len)
            - bert_label: padded label tensor of Size(max_len)
            - segment_label: 1s and 2s tensor of Size(max_len)
            - is_positive: bool, indicating whether is positive or not 
        '''
        t2_all_zero = True
        while t2_all_zero: #randomly mask at least one position in t2
            # uniformly sample unmasking rate t
            t = random.random()
            if t == 0:
                t = 0.05
            # get pos / neg sentence pair
            t1, t2 = self.get_sent(item, self.is_pos)
            # print(f'Inside __getitem__(), original t1: \n{t1}, \nt2: \n{t2}')
            # randomly mask t1 and t2 with a uniformly sampled masking rate t
            if self.stage == 'pretrain':
                (t1_processed, t1_label), (t2_processed, t2_label) = self.mask(t1, t), self.mask(t2, t) #tokenized
                t2_all_zero = all(x == 0 for x in t2_label)
            # mask t2 with random t while keep t1 unmasked
            elif self.stage == 'sft':
                (t1_processed, t1_label), (t2_processed, t2_label) = self.mask(t1, 0), self.mask(t2, t) #t
                t2_all_zero = all(x == 0 for x in t2_label)
            # keep t1 unmasked and t2 fully masked as the initial sequence 
            elif self.stage == 'inference':
                (t1_processed, t1_label), (t2_processed, t2_label) = self.mask(t1, 0), self.mask(t2, 1)
                t2_all_zero = False
            else:
                raise NotImplementedError
        
        # complete the start and end of sequences and their labels with special tokens
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_processed + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_processed + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]
        # print(f'After added special tokens, t1({len(t1)}): \n{t1}, \nt1_label: \n{t1_label}, \nt2({len(t2)}): {t2}, \nt2_label: {t2_label}')
        # concatenate t1 and t2 and add padding to max_len
        # print(f'\nself.max_len: {self.max_len}')
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.max_len]
        bert_input = (t1 + t2)[:self.max_len]
        bert_label = (t1_label + t2_label)[:self.max_len]
        padding = [self.tokenzier.vocab['[PAD]'] for _ in range(self.max_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        
        output = {
            'bert_input': bert_input,
            'bert_label': bert_label,
            'segment_label': segment_label,
        }
        # print(f'final:\nbert_input: \n{bert_input}, \nbert_label: \n{bert_label},\nsegment_label: \n{segment_label}')
        output = {k: torch.tensor(v) for k, v in output.items()}
        output['is_positive'] = self.is_pos
        
        return output
    
    def __getitem__pn(self, item):
        '''
        Select positive and negative sample pairs from the data_pair, and preprocess.  
        
        params:
            - item: line index
            - t: a random masking rate from a specific schedule
            - is_pos: positive means x and y from the same line pair; otherwise negative
            
        output_dict:
            - bert_input: masked context tensor of Size(max_len)
            - bert_label: padded label tensor of Size(max_len)
            - segment_label: 1s and 2s tensor of Size(max_len)
            - is_positive: bool, indicating whether is positive or not 
        '''
        t2_all_zero = True
        while t2_all_zero: #randomly mask at least one position in t2
            # uniformly sample unmasking rate t
            t = random.random()
            if t == 0:
                t = 0.05
            # get pos / neg sentence pair
            t1, pos_t2, neg_t2 = self.get_sents(item)
            # print(f'Inside __getitem__(), original t1: \n{t1}, \nt2: \n{t2}')
            # randomly mask t1 and t2 with a uniformly sampled masking rate t
            if self.stage == 'pretrain':
                (t1_processed, t1_label), (pos_t2_processed, pos_t2_label, \
                    neg_t2_processed, neg_t2_label) = self.mask(t1, t), \
                        self.mask_pn(pos_t2, neg_t2, t) # pos_t2 and neg_t2 should take the same masks
                t2_all_zero = (all(x == 0 for x in pos_t2_label) or all(x==0 for x in neg_t2_label))
            # mask t2 with random t while keep t1 unmasked
            elif self.stage == 'sft':
                (t1_processed, t1_label), (pos_t2_processed, pos_t2_label, \
                    neg_t2_processed, neg_t2_label) = self.mask(t1, 0), self.mask_pn(pos_t2, neg_t2, t)
                t2_all_zero = (all(x == 0 for x in pos_t2_label) or all(x==0 for x in neg_t2_label))
            # keep t1 unmasked and t2 fully masked as the initial sequence 
            elif self.stage == 'inference':
                t1_processed, t1_label, pos_t2_processed, pos_t2_label = self.mask(t1, 0), self.mask(t2, 1)
                t2_all_zero = False
            else:
                raise NotImplementedError
            
        pos_neg_outputs = []
        for is_neg, (t2_processed, t2_label) in enumerate(zip([pos_t2_processed, neg_t2_processed], \
            [pos_t2_label, neg_t2_label])):
            # complete the start and end of sequences and their labels with special tokens
            t1 = [self.tokenizer.vocab['[CLS]']] + t1_processed + [self.tokenizer.vocab['[SEP]']]
            t2 = t2_processed + [self.tokenizer.vocab['[SEP]']]
            t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
            t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]
            # print(f'After added special tokens, t1({len(t1)}): \n{t1}, \nt1_label: \n{t1_label}, \nt2({len(t2)}): {t2}, \nt2_label: {t2_label}')
            # concatenate t1 and t2 and add padding to max_len
            # print(f'\nself.max_len: {self.max_len}')
            segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.max_len]
            bert_input = (t1 + t2)[:self.max_len]
            bert_label = (t1_label + t2_label)[:self.max_len]
            padding = [self.tokenzier.vocab['[PAD]'] for _ in range(self.max_len - len(bert_input))]
            bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
            
            assert bert_label.count(0) != 0, f'Inside iter(), bert_label: \n{bert_label}, \npos_t2_label: \n{pos_t2_label}, \nneg_t2_label: \n{neg_t2_label}'
        
            output = {
                'bert_input': bert_input,
                'bert_label': bert_label,
                'segment_label': segment_label,
            }
            # print(f'final:\nbert_input: \n{bert_input}, \nbert_label: \n{bert_label},\nsegment_label: \n{segment_label}')
            output = {k: torch.tensor(v) for k, v in output.items()}
            output['is_positive'] = (1-is_neg)
            pos_neg_outputs.append(output)
        
        return pos_neg_outputs
    
    def get_sents(self, idx):
        t1, pos_t2, neg_t2 = self.lines[idx][0], self.lines[idx][1], \
            self.lines[rand.randrange(len(self.lines))][1]
        return t1, pos_t2, neg_t2
        
        
    def get_sent(self, idx, is_pos):
        if is_pos:
            t1, t2 = self.lines[idx][0], self.lines[idx][1]
        else:
            t1, t2 = self.lines[idx][0], self.lines[rand.randrange(len(self.lines))][1]
            
        return t1, t2
    
    def mask(self, sentence, t):
        tokens = sentence.split() #only useful for textual sentences
        output_label = []
        output =[]
        # t% of the tokens will be masked
        for i, token in enumerate(tokens): # iter through words
            prob = rand.random()
            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1] # token list
            # mask tokens of a word with t%
            if prob < t:
                for i in range(len(token_id)):
                    output.append(self.tokenizer.vocab['[MASK]'])
                output_label.append(token_id)
            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)
        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label
    
    def mask_pn(self, pos_seq, neg_seq, t):
        pos_tokens, neg_tokens = pos_seq.split(), neg_seq.split() #only useful for textual sentences
        pos_output_label, pos_output, neg_output_label, neg_output = [], [], [], []
        # t% of the tokens will be masked
        for i, (pos_token, neg_token) in enumerate(zip(pos_tokens, neg_tokens)): # iter through words
            prob = rand.random()
            # remove cls and sep token
            pos_token_id, neg_token_id = self.tokenizer(pos_token)['input_ids'][1:-1], \
                self.tokenizer(neg_token)['input_ids'][1:-1]# token list
            # mask tokens of a word with t%
            if prob < t:
                for i in range(len(pos_token_id)):
                    pos_output.append(self.tokenizer.vocab['[MASK]'])
                for i in range(len(neg_token_id)):
                    neg_output.append(self.tokenizer.vocab['[MASK]'])
                pos_output_label.append(pos_token_id)
                neg_output_label.append(neg_token_id)
            else:
                pos_output.append(pos_token_id)
                neg_output.append(neg_token_id)
                for i in range(len(pos_token_id)):
                    pos_output_label.append(0)
                for i in range(len(neg_token_id)):
                    neg_output_label.append(0)
        # flattening
        pos_output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in pos_output]))
        pos_output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in pos_output_label]))
        neg_output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in neg_output]))
        neg_output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in neg_output_label]))
        assert len(pos_output) == len(pos_output_label) == len(neg_output) == len(neg_output_label)
        assert pos_output.count(self.tokenizer.vocab['[MASK]']) == neg_output.count(self.tokenizer.vocab['[MASK]'])
        return pos_output, pos_output_label, neg_output, neg_output_label
        
    
### attention layers
class MultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0, f'a_model: {d_model}, heads: {heads}'
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)    

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    '''
    layer: Attn -> drop -> LN -> FFN -> drop -> LN
    '''
    def __init__(
        self, 
        d_model=768,
        heads=12, 
        feed_forward_hidden=768 * 4, 
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    
class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_hidden=128, n_layers=6, heads=6, max_len=256, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param d_hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.heads = heads
        # print(f'Init BERT: vocab_size: {vocab_size}, d_hidden: {d_hidden}, max_len: {max_len}, ') #########

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_hidden, seq_len=max_len) #只有这里用了max_len?

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_hidden, heads, d_hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info): #inputs are two tensors
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x    
    
class MLMDecoder(nn.Module): #task-specified decoder
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, out_len, vocab_size):
        """
        :param out_len: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(out_len, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, is_ebm=True): #inputs are output from BERT and an is_ebm flag
        # print(f'\ndecoder input\'s shape: {x.shape}') #Size(batch_size, seq_len, hidden_size)
        if is_ebm:
            return self.linear(x) #Size(batch_size, out_len, vocab_size) ?
        else:
            return self.softmax(self.linear(x))
        
'''The ultimate base_model for sequential EBMs'''
class DiscreteDiffusion(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, out_len=10, n_layers=6, heads=8, \
        max_len=256, dropout=0.1):
        super().__init__()
        self.bert = BERT(
            vocab_size=vocab_size, 
            d_hidden=hidden_size, # = d_model
            n_layers=n_layers, 
            heads=heads, 
            max_len=max_len,
            dropout=dropout
        )
        self.decoder = MLMDecoder(
            hidden_size, 
            vocab_size
        )
    def forward(self, x, segment_label, is_ebm):
        return self.decoder(self.bert(x, segment_label), is_ebm)
    
'''
Train a WordPiece Tokenizer
'''
def train_tokenizer(task):
    if task.startswith('binary'):
        paths = [str(x) for x in Path('../datasets/binary_arith_txt').glob('**/*.txt')]
        print(f'Total: {len(paths)} paths are found. \n{paths}')
    else:
        raise NotImplementedError
    
    ### training own tokenizer
    tokenizer = BertWordPieceTokenizer(
        clean_text=True, #remove control characters
        # handle_chinese_chars=False,
        # strip_accents=False,
        lowercase=True
    )

    if task.startswith('binary'):
        vocab_size=2
    elif task == 'sudoku':
        vocab_size=9
        
    tokenizer.train( 
        files=paths,
        vocab_size=vocab_size,  #0, 1 and special tokens
        # min_frequency=5,
        # limit_alphabet=1000, 
        # wordpieces_prefix='##',
        special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]', ' +++$+++ '],
        show_progress=True
        )

    tokenizer.save_model('./models', f'tokenizer_{task}')
    # tokenizer = BertTokenizer.from_pretrained(f'./models/tokenizer_{task}-vocab.txt', local_files_only=True)
    print(f'\nTrained tokenizer saved to "./models/tokenizer_{task}"')
    return tokenizer
    
# if __name__ == "__main__":
#     train_tokenizer('binary')