from constants import VOCAB_SIZE, MAX_LENGTH
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
class Engine:
    def __init__(self, args: Dict[str, Any]):
        # engine attributes:
        # encoder/decoder_optim
        # encoder/decoder
        # device
        lr = float(args['--lr'])
        hs = int(args['--hidden_size'])
        os = int(args['--output_size'])
        dropout = float(args['--dropout'])

        self.device = device(args['--device'])

        self.encoder = EncoderRNN(input_size=VOCAB_SIZE,
                                       hidden_size=hs,
                                       device=self.device)

        self.decoder = AttnDecoderRNN(hidden_size=hs,
                                           output_size=os,
                                           dropout_p=dropout,
                                           device=self.device)

        self.encoder_optim = optim.Adam(self.encoder.parameters(), lr=lr)
        self.decoder_optim = optim.Adam(self.decoder.parameters(), lr=lr)
        

class ProteinEncoderRNN(nn.Module):
    def __init__(self):
        pass

class EncoderRNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.hidden = self.initHidden()

    def _forward(self, x, data="none") -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x).view(1, 1, -1)

        try: output, self.hidden = self.gru(embedded, self.hidden)
        except: output, self.hidden = self.gru(embedded, self.hidden)

        return output, self.hidden

    def initHidden(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

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

        self.hidden = self.initHidden()

    def _forward(self, x:torch.Tensor, encoder_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)

        try: attn_weights = F.softmax(self.attn(torch.cat((embedded[0], self.hidden[0]), 1)), dim=1)
        except: attn_weights = F.softmax(self.attn(torch.cat((embedded[0], self.hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, self.hidden = self.gru(output, self.hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, attn_weights

    def initHidden(self) -> torch.Tensor:
        self.hidden = torch.zeros(1, 1, self.hidden_size).to(self.device)
        return self.hidden

    def setHidden(self, hidden) -> None:
        self.hidden = hidden


