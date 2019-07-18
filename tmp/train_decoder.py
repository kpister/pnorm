"""Train Decoder Architecture
Usage: train_decoder.py [options]

Options:
    --epochs INT
    --lr FLOAT
    --device STR
    --hidden_size INT
    --output_size INT
    --dropout FLOAT
    -pt, --protein_training FILE
    -pv, --protein_validation FILE
    -md, --morphology_data FILE
"""

# personal libs
from constants import SOS_token, EOS_token, MAX_LENGTH
import models
import loss 
import dataset

# other libs
import torch 
from torch.autograd import Variable
from docopt import docopt #type: ignore
from typing import Dict, Any
import random

def train(engine: models.Engine, 
          prot_data: dataset.ProteinData, 
          morph_data: dataset.MorphemeData):

    engine.encoder.train()
    engine.decoder.train()
    prot_epoch_loss = 0
    morph_epoch_loss = 0

    for idx in range(max(len(prot_data), len(morph_data))):
        total_loss = None

        engine.encoder_optim.zero_grad()
        engine.decoder_optim.zero_grad()

        # Train PEN
        if idx < len(prot_data):
            x, y = zip(*prot_data[idx])

            x_embedding = engine.encoder._forward(x, data='protein')
            y_embedding = engine.encoder._forward(y, data='protein')
            prot_loss = loss.SimilarityLoss(x_embedding, y_embedding)
            total_loss = prot_loss

            prot_epoch_loss += prot_loss.item()

        # Train Morphology, adapted from:
        # https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
        if idx < len(morph_data):
            # single morpheme per batch
            lemma, tag, word = morph_data[idx]

            # Encoder morpheme
            engine.encoder.hidden = engine.encoder.initHidden()

            input_tensor = torch.cat((word[0], tag[0]))
            target_length = lemma[0].size(0)

            encoder_outputs = torch.zeros(MAX_LENGTH, engine.encoder.hidden_size, device=engine.device)

            meme_loss = loss.InitMorphemeLoss()

            encoder_output, encoder_hidden = engine.encoder._forward(input_tensor)
            encoder_outputs[:input_tensor.size(0)] = encoder_output[0] # 1 x 1 x |E|

            # Decode morpheme
            decoder_input = torch.tensor([[SOS_token]], device=engine.device)

            engine.decoder.setHidden(encoder_hidden)

            use_teacher_forcing = True if random.random() < 0.5 else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_attention = engine.decoder._forward(
                        decoder_input, encoder_outputs)
                    meme_loss += loss.MorphemeLoss(decoder_output, lemma[0][di])
                    decoder_input = lemma[0][di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_attention = engine.decoder._forward(
                        decoder_input, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    meme_loss += loss.MorphemeLoss(decoder_output, lemma[0][di])
                    if decoder_input.item() == EOS_token:
                        break

            if total_loss:
                total_loss += meme_loss
            else:
                total_loss = meme_loss

            morph_epoch_loss += meme_loss.item()

        total_loss.backward() # type: ignore

        # Clip gradient norms
        clip = 0.1
        torch.nn.utils.clip_grad_norm(engine.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(engine.decoder.parameters(), clip)
        engine.encoder_optim.step()
        engine.decoder_optim.step() 

    return (prot_epoch_loss, morph_epoch_loss)

if __name__ == '__main__':
    args = docopt(__doc__)
    epochs = int(args['--epochs'])

    prot_training_data = dataset.ProteinData(args['--protein_training'])
    prot_val_data      = dataset.ProteinData(args['--protein_validation'])
    morph_data         = dataset.MorphemeData(args['--morphology_data']) 

    engine = models.Engine(args)

    for e in range(epochs):
        loss = train(engine=engine, 
                     prot_data=prot_training_data, 
                     morph_data=morph_data)

