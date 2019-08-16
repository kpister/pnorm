from constants import VOCAB_SIZE, MAX_LENGTH

import loss
import encoder
import seq2seq as ss

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

from torch import optim, device
from typing import Dict, Any, List, Tuple
import math

# Return engine dictionary
# Args expects:
# --device: cuda, cpu, cuda:0...
# --lr: 0.1, 0.001...
# --hidden_size: 200...
# --output_size: 200...
# --dropout: 0.1, 0.2...
# --margin: 2.0, ...
# --char_embedding 200...
# --word_embedding 300...
# --layers 3,4...
class Engine:
    def __init__(self, args: Dict[str, Any]):
        # engine attributes:
        # encoder/decoder_optim
        # encoder/decoder
        # device
        # protein/morpheme_criterion
        in_dim = VOCAB_SIZE
        ce = int(args['--char_embedding'])
        we = int(args['--word_embedding'])
        layers = int(args['--layers'])
        lr = float(args['--lr'])
        hs = int(args['--hidden_size'])
        os = int(args['--output_size'])
        dropout = float(args['--dropout'])
        margin = float(args['--margin'])

        self.alpha = float(args['--alpha'])
        self.device = device(args['--device'])
        self.teacher_forcing = float(args['--teacher_forcing'])

        self.encoder = encoder.EncoderLstm({
            'char_embedding': ce,
            'device': self.device,
            'layers': layers,
            'hidden': hs,
            'bidirectional': True,
            'encoder': 'lstm',
            'activation': 'relu',
            'input_dim': in_dim,
            'dropout': dropout,
            'word_embedding': we
            })

        if args['--protein_data'] != '':
            self.pEncoder = self.encoder
            if args['--load']:
                self.pEncoder.load_state_dict(torch.load(f"files/{args['--load']}.protein.pkl", map_location=self.device))

            self.pEncoder = self.pEncoder.to(self.device)
            self.pLoss = loss.SimilarityLoss(margin=margin)
            self.pOptimizer = optim.Adam(self.encoder.parameters(), lr=lr)

        if args['--morpheme_data'] != '':
            morpheme_decoder = ss.Decoder(embed_size=we, hidden_size=hs, output_size=in_dim)
            self.mSeq2Seq = ss.Seq2Seq(self.encoder, morpheme_decoder, self.device)
            if args['--load']:
                self.mSeq2Seq.load_state_dict(torch.load(f"files/{args['--load']}.morpheme.pkl", map_location=self.device))

            self.mSeq2Seq.to(self.device)
            self.mOptimizer = optim.Adam(self.mSeq2Seq.parameters(), lr=lr)

        #self.acro_decoder = AttnDecoderRNN(embedding_size=we, hidden_size=hs, vocab_size=in_dim, dropout_p=dropout, device=self.device)
        #if args['--load']:
            #self.acro_decoder.load_state_dict(torch.load(f"files/{args['--load']}.adecoder.pkl", map_location=self.device))

        #self.para_decoder = AttnDecoderRNN(embedding_size=we, hidden_size=hs, vocab_size=in_dim, dropout_p=dropout, device=self.device)
        #if args['--load']:
            #self.para_decoder.load_state_dict(torch.load(f"files/{args['--load']}.adecoder.pkl", map_location=self.device))

 
        #self.acro_decoder.to(self.device)
        #self.para_decoder.to(self.device)

        #self.acro_decoder_optim = optim.Adam(self.acro_decoder.parameters(), lr=lr)
        #self.para_decoder_optim = optim.Adam(self.para_decoder.parameters(), lr=lr)

        #self.morpheme_criterion = loss.MorphemeLoss()

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(1)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        #encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, dropout_p, device, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.n_layers = 1
        self.embed_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = Attention(self.hidden_size)
        self.attn = nn.Linear(self.hidden_size + embedding_size, max_length)
        self.attn_combine = nn.Linear(self.hidden_size + embedding_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, x:torch.Tensor, hidden:torch.Tensor, encoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x) # 1 x batch_size x embedding_dim
        embedded = self.dropout(embedded)

        attn_weights = self.attention(hidden[-1], encoder_outputs)

        #try: attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        #except: attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        # |attn_weights| = batch_size x MAX_LENGTH

        # bmm: b x n x m `@` b x m x p
        # aw :: b x 1 x MAX_LEN 
        # eo :: b x MAX_LEN x Embedding
        #attn_applied = attn_weights.unsqueeze(1).bmm(encoder_outputs).permute(1,0,2)
        attn_applied = attn_weights.bmm(encoder_outputs).permute(1,0,2)

        output = torch.cat((embedded, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden) #ERROR

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size).to(self.device)
