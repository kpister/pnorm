import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import math

vocab_size = 128
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0
    return X

def lineToTensor(line):
    tensor = torch.zeros(len(line), vocab_size, dtype=torch.float)
    for li, letter in enumerate(line):
        tensor[li][ord(letter)] = 1
    return tensor

class Siamese(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(Siamese, self).__init__()

        self.num_layers = 1
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # use non-linearity function after linear (MLP)
        self.fc = nn.Sequential(
                nn.Linear(hidden_dim*2, 100),
                nn.ReLU())
        self.device = device

    def init_hidden(self, batch):
        return (Variable(torch.randn(2*self.num_layers, batch, self.hidden_dim)).to(self.device),
                Variable(torch.randn(2*self.num_layers, batch, self.hidden_dim)).to(self.device))


    # use satellite numbers to sort and then resort
    def forward_one(self, x, hidden):
        text_ = sorted([[p, i] for i, p in enumerate(x)], 
                        key=lambda v:len(v[0]), reverse=True)

        x_ = self.pad_and_pack_batch(text_).to(self.device)
        lstm_out, _ = self.lstm(x_, hidden)

        # unpack sequences
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Use lstm_out average
        out = torch.mean(out, 1)

        # cuda 9.2 error
        try:
            x_ = self.fc(out)
        except:
            x_ = self.fc(out)
        x_ = l2norm(x_)
        if len([i for i,v in enumerate(x_) if math.isnan(v[0])]):
            import pdb;pdb.set_trace()

        # resort
        for i, v in enumerate(x_):
            text_[i][0] = v

        return torch.stack([i[0] for i in sorted(text_, key=lambda v:v[1])])

    def pad_and_pack_batch(self, text):
        # size: batch_size
        seq_lengths = torch.LongTensor(list(map(lambda x: len(x[0]), text))).to(self.device)
        
        # size: batch_size x longest_seq x vocab_size
        seq_tensor = Variable(torch.zeros((len(text), seq_lengths.max(), vocab_size)))

        for idx, seqlen in enumerate(seq_lengths):
            seq_tensor[idx, :seqlen, :] = text[idx][0]

        # size = longest_seq x batch_size x vocab_size
        return pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)

    # x1, x2 are batches
    def forward(self, x1, x2):
        self.hidden = self.init_hidden(len(x1))
        out1 = self.forward_one(x1, self.hidden)
        out2 = self.forward_one(x2, self.hidden)

        return out1, out2

    def batchless(self, x1, x2):
        return self.forward([x1], [x2])

    def batchless_one(self, x):
        hidden = self.init_hidden(1)
        lstm_out, _ = self.lstm(torch.stack([lineToTensor(x).to(self.device)]), hidden)

        # Use lstm_out average
        out = torch.mean(lstm_out, 1)

        # cuda 9.2 error
        try:
            return self.fc(out)
        except:
            return self.fc(out)

# inspiration: https://github.com/fangpin/siamese-pytorch/blob/master/model.py
