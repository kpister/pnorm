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
    def __init__(self, input_dim, char_embedding_dim, hidden_dim, output_dim, dropout_rate, num_layers, device):
        super(Siamese, self).__init__()

        self.char_embedding_dim = char_embedding_dim
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = True

        self.char_embed = nn.Embedding(input_dim, char_embedding_dim)
        self.lstm = nn.LSTM(char_embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=self.bidirectional)
        
        self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim*(1+self.bidirectional), output_dim),
                nn.ReLU())

    def init_hidden(self, batch):
        return (Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device),
                Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device))


    def forward_one(self, x, hidden):
        # use satellite numbers to sort and then resort
        sat = sorted([[self.char_embed(p.to(self.device)),i] for i, p in enumerate(x)],
                       key=lambda v:v[0].shape[0], reverse=True)

        # encode
        lstm_out, _ = self.lstm(self.pad_and_pack_batch(sat).to(self.device), hidden)

        # unpack sequences
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Use lstm_out average or max pooling?
        out = torch.mean(out, 1)
        #out = torch.argmax(out, 1)

        # cuda 9.2 error
        try: x_ = self.fc(out)
        except: x_ = self.fc(out)

        # when training with cosine similarity
        #x_ = l2norm(x_)

        # resort
        for i, v in enumerate(x_):
            sat[i][0] = v

        return torch.stack([i[0] for i in sorted(sat, key=lambda v:v[1])])

    def pad_and_pack_batch(self, sat):
        # size: batch_size
        seq_lengths = torch.LongTensor(list(map(lambda x: len(x[0]), sat))).to(self.device)
        
        # size: batch_size x longest_seq x vocab_size
        seq_tensor = Variable(torch.zeros((len(sat), seq_lengths.max(), self.char_embedding_dim))).to(self.device)

        for idx, seqlen in enumerate(seq_lengths):
            seq_tensor[idx, :seqlen, :] = sat[idx][0]

        # size = longest_seq x batch_size x vocab_size
        return pack_padded_sequence(seq_tensor, seq_lengths, batch_first=True)

    # x, y are batches
    def forward(self, x, y):
        assert len(x) == len(y)
        hidden = self.init_hidden(len(x))
        return self.forward_one(x, hidden), self.forward_one(y, hidden)

