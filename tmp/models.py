from constants import VOCAB_SIZE, MAX_LENGTH
import loss
import encoder

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore

from torch import optim, device
from typing import Dict, Any, List, Tuple

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
        in_dim = int(args['--input_dim'])
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
        
        if args['--load']:
            self.encoder.load_state_dict(torch.load('best_model.pkl', map_location=self.device))

        self.decoder = AttnDecoderRNN(hidden_size=we,
                                           output_size=in_dim,
                                           dropout_p=dropout,
                                           device=self.device)
        try:
            self.encoder.to(self.device)
            self.decoder.to(self.device)
        except:
            self.encoder.to(self.device)
            self.decoder.to(self.device)

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=lr)

        self.protein_criterion = loss.SimilarityLoss(margin=margin)
        self.morpheme_criterion = loss.MorphemeLoss()
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p, device, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def _forward(self, x:torch.Tensor, hidden:torch.Tensor, encoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x) # 1 x batch_size x embedding_dim
        embedded = self.dropout(embedded)

        try: attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        except: attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 1)), dim=1)
        # |attn_weights| = batch_size x MAX_LENGTH

        # bmm: b x n x m `@` b x m x p
        # aw :: b x 1 x MAX_LEN 
        # eo :: b x MAX_LEN x Embedding
        attn_applied = attn_weights.unsqueeze(1).bmm(encoder_outputs).permute(1,0,2)

        output = torch.cat((embedded, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        #output, hidden = self.gru(output, hidden) #ERROR

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size).to(self.device)
