"""Train Decoder Architecture
Usage: 
    train_decoder.py [options]

Options:
    -h, --help
    --alpha FLOAT                       set the ratio of losses [default: 0.16]
    --epochs INT                        set number of epochs [default: 20]
    --lr FLOAT                          set the learning rate [default: 0.001]
    --device STR                        set the device [default: cuda:0]
    --hidden_size INT                   set the size of encoder hidden [default: 200]
    --output_size INT                   set the output size of the decoder [default: 200]
    --dropout FLOAT                     set the dropout rate [default: 0.3]
    --margin FLOAT                      set the loss margin [default: 2.0]
    --char_embedding INT                set the size of char embedding [default: 200]
    --word_embedding INT                set the size of word embedding [default: 200]
    --layers INT                        set the depth of the encoder [default: 5]
    --batch_size INT                    set the batch size [default: 500]
    --load                              load a model file [default: False]
    -t, --protein_training FILE         set the input file
    -v, --protein_validation FILE       set the other input file
    -d, --morphology_data FILE          set the last input file
"""

# personal libs
from constants import *
import models
import dataset

# other libs
from docopt import docopt #type: ignore
import torch 
from torch.autograd import Variable
from typing import Dict, Any
import random
from tqdm import trange #type: ignore

def train(engine: models.Engine, 
          prot_data: dataset.ProteinData, 
          morph_data: dataset.MorphemeData):

    engine.encoder.train()
    engine.decoder.train()
    prot_epoch_loss:float = 0
    morph_epoch_loss:float = 0
    torch.autograd.set_detect_anomaly(True)

    for idx in trange(max(len(prot_data), len(morph_data))):
        total_loss = torch.tensor([0], dtype=torch.float, device=engine.device)

        engine.encoder_optim.zero_grad()
        engine.decoder_optim.zero_grad()

        # Train PEN
        if idx < len(prot_data):
            x, y = zip(*prot_data[idx])
            enc_hidden = engine.encoder.initHidden(batch_size=len(x))

            x_embedding, _hs = engine.encoder._forward(x, enc_hidden, data='protein') #type: ignore
            y_embedding, _hs = engine.encoder._forward(y, enc_hidden, data='protein') #type: ignore
            prot_loss = engine.protein_criterion._forward(x_embedding, y_embedding)
            total_loss += prot_loss.view(1).to(engine.device)

            prot_epoch_loss += prot_loss.item() / len(x)
            enc_hidden = _hs

        # Train Morphology, adapted from:
        # https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
        if idx < len(morph_data):
            # single morpheme per batch
            meme_loss = torch.tensor([0], dtype=torch.float, device=engine.device)

            lemma, input_tensor = zip(*morph_data[idx])
            lemma_tensor = torch.stack(lemma).transpose(0,1).to(engine.device)

            enc_out, enc_hidden = engine.encoder._forward_unordered( #type:ignore
                    input_tensor, hidden=enc_hidden)

            # Decode morpheme
            dec_input = torch.tensor([SOS_token] * len(lemma), device=engine.device)

            dec_hidden = enc_hidden[0][-1]

            use_teacher_forcing = True if random.random() < 1.0 else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(MAX_LENGTH):
                    dec_output, dec_hidden, _ = engine.decoder._forward(
                        dec_input, dec_hidden, enc_out)
                    meme_loss += engine.morpheme_criterion(dec_output, lemma_tensor[di])
                    dec_input = lemma_tensor[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(MAX_LENGTH):
                    dec_output, dec_hidden, _ = engine.decoder._forward(
                        dec_input, dec_hidden, enc_out)
                    topv, topi = dec_output.topk(1)
                    dec_input = topi.squeeze().detach()  # detach from history as input

                    meme_loss += engine.morpheme_criterion(dec_output, lemma_tensor[di])
                    if dec_input.item() == EOS_token:
                        break

            total_loss += meme_loss / morph_data.batch_size * engine.alpha
            morph_epoch_loss += meme_loss.item() / morph_data.batch_size

        total_loss.backward(retain_graph=True)

        engine.encoder_optim.step()
        engine.decoder_optim.step() 

    return (prot_epoch_loss, morph_epoch_loss)

def evaluate(engine, prot_data):
    engine.encoder.eval()
    loss = 0

    with torch.no_grad():
        for idx in trange(len(prot_data)):
            x, y = zip(*prot_data[idx])
            enc_hidden = engine.encoder.initHidden(batch_size=len(x))

            x_embedding, _hs = engine.encoder._forward(x, enc_hidden, data='protein') #type: ignore
            y_embedding, _hs = engine.encoder._forward(y, enc_hidden, data='protein') #type: ignore
 
            loss += engine.protein_criterion._forward(x_embedding, y_embedding).item()
    return loss / len(prot_data)


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    epochs = int(args['--epochs'])
    bs = int(args['--batch_size'])

    prot_training_data = dataset.ProteinData(args['--protein_training'], batch_size=bs)
    morph_data         = dataset.MorphemeData(args['--morphology_data'], batch_size=bs) 
    args['--input_dim'] = VOCAB_SIZE + morph_data.num_tags

    if args['--protein_validation']:
        prot_val_data  = dataset.ProteinData(args['--protein_validation'])

    engine = models.Engine(args)

    best_vloss = 10

    for e in range(epochs):
        loss = train(engine=engine, 
                     prot_data=prot_training_data, 
                     morph_data=morph_data)
        vloss = evaluate(engine, prot_val_data)
        if vloss < best_vloss:
            torch.save(engine.encoder.state_dict(), 'best_model.pkl')

        print(f'Epoch {e} complete. Loss: {loss[0]:.4f}, {loss[1]:.4f}\tValidation Loss: {vloss:.4f}')

