import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0 # remove nan
    return X

# siamese network inspiration: https://github.com/fangpin/siamese-pytorch/blob/master/model.py
class Siamese(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(Siamese, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = True
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)
        # use non-linearity function after linear (MLP)
        self.fc = nn.Sequential(
                nn.Linear(hidden_dim*(1+self.bidirectional), output_dim),
                nn.ReLU())

    def init_hidden(self, batch):
        return (Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device),
                Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device))


    def forward_one(self, x, hidden):
        # use satellite numbers to sort and then resort
        text_ = sorted([[p, i] for i, p in enumerate(x)], 
                        key=lambda v:len(v[0]), reverse=True)

        x_ = self.pad_and_pack_batch(text_).to(self.device)
        lstm_out, _ = self.lstm(x_, hidden)

        # unpack sequences
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Use lstm_out average or TODO: attention?
        out = torch.mean(out, 1)

        # cuda 9.2 error
        try:
            x_ = self.fc(out)
        except:
            x_ = self.fc(out)
        #x_ = l2norm(x_)

        # resort
        for i, v in enumerate(x_):
            text_[i][0] = v

        return torch.stack([i[0] for i in sorted(text_, key=lambda v:v[1])])

    def pad_and_pack_batch(self, text):
        # size: batch_size
        seq_lengths = torch.LongTensor(list(map(lambda x: len(x[0]), text))).to(self.device)
        
        # size: batch_size x longest_seq x vocab_size
        seq_tensor = Variable(torch.zeros((len(text), seq_lengths.max(), self.input_dim))).to(self.device)

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

class CharEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, device):
        super(CharEncoder, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.char_enc = nn.LSTM(input_dim, output_dim, self.num_layers, batch_first=True)

    def init_hidden(self, batch):
        return (Variable(torch.randn(self.num_layers, batch, self.output_dim)).to(self.device),
                Variable(torch.randn(self.num_layers, batch, self.output_dim)).to(self.device))

    def forward(self, x):
        #shape of x: (batch_size, vocab_size)
        x = x.view(1, *x.size()).to(self.device)
        hidden = self.init_hidden(x.size(0))

        #output shape: (batch_size, output_dim)
        enc, _ = self.char_enc(x, hidden)
        return enc[0]
