import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import List, Tuple, Any, Dict

# siamese network inspiration: https://github.com/fangpin/siamese-pytorch/blob/master/model.py
class EncoderLstm(nn.Module):
    def __init__(self, opts:Dict[str, Any]) -> None:
        super(EncoderLstm, self).__init__()

        self.char_embedding_dim = opts['char_embedding']
        self.device = opts['device']
        self.num_layers = opts['layers']
        self.hidden_dim = opts['hidden']
        self.bidirectional = opts['bidirectional']
        self.encoder = opts['encoder']
        non_linear_activation = nn.ReLU() if opts['activation'] == 'relu' else nn.Tanh()

        self.char_embed = nn.Embedding(opts['input_dim'], self.char_embedding_dim)

        #self.prot_embed = nn.GRU(self.char_embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=opts['dropout'])
        self.prot_embed = nn.LSTM(self.char_embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=opts['dropout'])
        
        self.fc = nn.Sequential(
                nn.Dropout(opts['dropout']),
                nn.Linear(self.hidden_dim*(1+self.bidirectional), opts['word_embedding']),
                non_linear_activation)

    def initHidden(self, batch_size):
        # lstm
        return (Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch_size, self.hidden_dim)).to(self.device),
                Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch_size, self.hidden_dim)).to(self.device))

        # gru
        #return Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)).to(self.device)

    def _forward_unordered(self, x:List[torch.Tensor], hidden:Tuple[torch.Tensor, torch.Tensor], data='None') -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sat = [self.char_embed(p.to(self.device)) for p in x]
        packed = self.pad_and_pack_batch_u(sat)
        output, hidden = self.prot_embed(packed.to(self.device), hidden)
        outputs, output_lens = pad_packed_sequence(output,batch_first=True)
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, : ,self.hidden_dim:]
        return outputs, hidden


    def _forward(self, x:List[torch.Tensor], hidden:Tuple[torch.Tensor, torch.Tensor], data='None') -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # use satellite numbers to sort and then resort
        sat = sorted([[self.char_embed(p.to(self.device)),i] for i, p in enumerate(x)],
                       key=lambda v:v[0].shape[0], reverse=True)

        # encode
        encoder_out, hs = self.prot_embed(self.pad_and_pack_batch(sat).to(self.device), hidden)

        # unpack sequences
        out, _ = pad_packed_sequence(encoder_out, batch_first=True)

        # Use lstm_out average or max pooling?
        out = torch.mean(out, 1)
        #out = torch.argmax(out, 1)

        # cuda 9.2 error
        try: x_ = self.fc(out)
        except: x_ = self.fc(out)

        # resort
        for i, v in enumerate(x_):
            sat[i][0] = v

        return torch.stack([i[0] for i in sorted(sat, key=lambda v:v[1])]), hs

    def pad_and_pack_batch_u(self, sat):
        # size: batch_size
        seq_lengths = torch.LongTensor(list(map(len, sat))).to(self.device)
        
        # size: batch_size x longest_seq x vocab_size
        seq_tensor = Variable(torch.zeros((len(sat), seq_lengths.max(), self.char_embedding_dim))).to(self.device)

        for idx, seqlen in enumerate(seq_lengths):
            seq_tensor[idx, :seqlen, :] = sat[idx]

        # size = longest_seq x batch_size x vocab_size
        return pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True, enforce_sorted=False)

    def pad_and_pack_batch(self, sat):
        # size: batch_size
        seq_lengths = torch.LongTensor(list(map(lambda x: len(x[0]), sat))).to(self.device)
        
        # size: batch_size x longest_seq x vocab_size
        seq_tensor = Variable(torch.zeros((len(sat), seq_lengths.max(), self.char_embedding_dim))).to(self.device)

        for idx, seqlen in enumerate(seq_lengths):
            seq_tensor[idx, :seqlen, :] = sat[idx][0]

        # size = longest_seq x batch_size x vocab_size
        return pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)

    def ppb(self, data):
        seq_lengths = torch.LongTensor(list(map(len, sat))).to(self.device)
        seq_lengths, sorted_indices = torch.sort(seq_lengths, descending=True)
        sorted_indices = sorted_indices.to(input.device)
        padded_data = pad_sequence(data, batch_first=True)
        sorted_data = padded_data.index_select(0, sorted_indices)

        return pack_padded_sequence(sorted_data, seq_lengths, batch_first=True), sorted_indices

    # si = sorted_indices
    def undo(self, data, si):
        if si is None:
            return None
        output = torch.empty_like(si)
        output.scatter_(0, si, torch.arange(0, si.numel(), device=si.device))
        return data.index_select(0, output)
