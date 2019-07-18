import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# based on protonet model
# in_dim = char_embedding_dim
def create_cnn(in_dim, hid_dim, out_dim):
    def conv_block(in_channels, out_channels):
        kernel = 3
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    encoder = nn.Sequential(
        conv_block(in_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, out_dim)
    )

    return encoder

# siamese network inspiration: https://github.com/fangpin/siamese-pytorch/blob/master/model.py
class Siamese(nn.Module):
    def __init__(self, opts):
        super(Siamese, self).__init__()

        self.char_embedding_dim = opts['char_embedding']
        self.device = opts['device']
        self.num_layers = opts['layers']
        self.hidden_dim = opts['hidden']
        self.bidirectional = opts['bidirectional']
        self.encoder = opts['encoder']
        non_linear_activation = nn.ReLU() if opts['activation'] == 'relu' else nn.Tanh()

        self.char_embed = nn.Embedding(opts['input_dim'], self.char_embedding_dim)

        if opts['encoder'] == 'cnn':
            self.prot_embed = create_cnn(self.char_embedding_dim, self.hidden_dim, self.hidden_dim)
        elif opts['encoder'] == 'gru':
            self.prot_embed = nn.GRU(self.char_embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, dropout=opts['dropout'])
        elif opts['encoder'] == 'lstm':
            self.prot_embed = nn.LSTM(self.char_embedding_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=opts['dropout'])
        
        self.fc = nn.Sequential(
                nn.Dropout(opts['dropout']),
                nn.Linear(self.hidden_dim*(1+self.bidirectional), opts['word_embedding']),
                non_linear_activation)

    def init_hidden(self, batch):
        if self.encoder == 'lstm':
            return (Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device),
                    Variable(torch.randn((1+self.bidirectional)*self.num_layers, batch, self.hidden_dim)).to(self.device))
        else:
            return Variable(torch.randn(self.num_layers, batch, self.hidden_dim)).to(self.device)


    def forward_one(self, x, hidden):
        # use satellite numbers to sort and then resort
        sat = sorted([[self.char_embed(p.to(self.device)),i] for i, p in enumerate(x)],
                       key=lambda v:v[0].shape[0], reverse=True)

        # encode
        encoder_out, _ = self.prot_embed(self.pad_and_pack_batch(sat).to(self.device), hidden)

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

