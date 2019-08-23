import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from constants import MAX_LENGTH
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
        self.mhs = opts['morph_out']
        non_linear_activation = nn.ReLU() if opts['activation'] == 'relu' else nn.Tanh()

        self.char_embed = nn.Embedding(opts['input_dim'], self.char_embedding_dim)

        #self.prot_embed = nn.GRU(self.char_embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=opts['dropout'])
        self.share_lstm = nn.LSTM(self.char_embedding_dim, self.mhs, batch_first=True)
        self.prot_embed = nn.LSTM(self.mhs*2, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=opts['dropout'])
        
        self.fc = nn.Sequential(
                nn.Dropout(opts['dropout']),
                nn.Linear(self.hidden_dim, opts['word_embedding']),
                non_linear_activation)

    def initHidden(self, batch_size):
        # lstm
        return (Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch_size, self.hidden_dim)).to(self.device),
                Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch_size, self.hidden_dim)).to(self.device))

        # gru
        #return Variable(torch.randn(self.num_layers, batch_size, self.hidden_dim)).to(self.device)

    # Forward for traditional embedding (ordered in descending length)
    def forward(self, x:torch.Tensor, x_len, hidden:Tuple[torch.Tensor, torch.Tensor], norm=False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        packed = pack_padded_sequence(self.char_embed(x), x_len, batch_first=True)
        output, _ = self.share_lstm(packed)
        output, hidden = self.prot_embed(output, hidden)
        outputs, _ = pad_packed_sequence(output,batch_first=True)
        # sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, : ,self.hidden_dim:]
        if norm:
            outputs = self.fc(torch.mean(outputs, 1))
        return outputs, hidden, None
