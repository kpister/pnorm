from constants import VOCAB_SIZE

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
    def __init__(self, args: Dict[str, Any]) -> None:
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
            #self.acro_decoder.load_state_dict(torch.load(f"files/{args['--load']}.acronym.pkl", map_location=self.device))

        #self.para_decoder = AttnDecoderRNN(embedding_size=we, hidden_size=hs, vocab_size=in_dim, dropout_p=dropout, device=self.device)
        #if args['--load']:
            #self.para_decoder.load_state_dict(torch.load(f"files/{args['--load']}.paraphrase.pkl", map_location=self.device))

 
        #self.acro_decoder.to(self.device)
        #self.para_decoder.to(self.device)

        #self.acro_decoder_optim = optim.Adam(self.acro_decoder.parameters(), lr=lr)
        #self.para_decoder_optim = optim.Adam(self.para_decoder.parameters(), lr=lr)

        #self.morpheme_criterion = loss.MorphemeLoss()
