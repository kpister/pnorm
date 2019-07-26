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
    --load FILE                         load a model file 
    --teacher_forcing FLOAT             set the teacher forcing ration [default: 0.5]
    -p, --protein_data PATH             set the input file [default: '']
    -m, --morphology_data FILE          set the morpheme input file
    -a, --acronym_data FILE             set the acronym data file
    --no_morphemes                      turn off the morpheme training [default: False]
    --no_acronyms                       turn off the acronyms training [default: False]
    --no_proteins                       turn off the proteins training [default: False]
"""

# personal libs
from constants import *
import evaluate
import models
import dataset

# other libs
import os
from typing import Dict, Any
import random
import torch 
from torch.autograd import Variable
from tqdm import trange #type: ignore
from docopt import docopt #type: ignore
import hashlib
import json

def train(engine: models.Engine, 
          prot_data: dataset.ProteinData, 
          morph_data: dataset.MorphemeData,
          acro_data: dataset.AcronymData):

    engine.encoder.train()
    engine.decoder.train()
    prot_epoch_loss:float = 0
    morph_epoch_loss:float = 0
    acro_epoch_loss:float = 0
    torch.autograd.set_detect_anomaly(True)

    for idx in trange(max(len(prot_data), len(morph_data), len(acro_data))):
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

        # Train Morphology, adapted from:
        # https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py
        if idx < len(morph_data):
            meme_loss = torch.tensor([0], dtype=torch.float, device=engine.device)

            lemma, input_tensor = zip(*morph_data[idx])
            lemma_tensor = torch.stack(lemma).transpose(0,1).to(engine.device)

            enc_hidden = engine.encoder.initHidden(batch_size=len(lemma))
            enc_out, enc_hidden = engine.encoder._forward_sorted( #type:ignore
                    input_tensor, hidden=enc_hidden)

            # Decode morpheme
            dec_input = torch.tensor([SOS_token] * len(lemma), device=engine.device)
            dec_hidden = enc_hidden[0][-1]

            tf = random.random() < engine.teacher_forcing
            # Teacher forcing: Feed the target as the next input
            for di in range(MAX_LENGTH):
                dec_output, dec_hidden, _ = engine.decoder._forward(dec_input, dec_hidden, enc_out)
                meme_loss += engine.morpheme_criterion(dec_output, lemma_tensor[di])

                if tf:
                    dec_input = lemma_tensor[di]  # Teacher forcing
                else:
                    topv, topi = dec_output.topk(1)
                    dec_input = topi.squeeze().detach()  # detach from history as input

            total_loss += meme_loss / morph_data.batch_size * engine.alpha
            morph_epoch_loss += meme_loss.item() / morph_data.batch_size

        if idx < len(acro_data):
            acro_loss = torch.tensor([0], dtype=torch.float, device=engine.device)

            acronym, expansion = zip(*acro_data[idx])
            acro_tensor = torch.stack(acronym).transpose(0,1).to(engine.device)

            enc_hidden = engine.encoder.initHidden(batch_size=len(acronym))
            enc_out, enc_hidden = engine.encoder._forward_sorted( #type:ignore
                    expansion, hidden=enc_hidden)

            # Decode morpheme
            dec_input = torch.tensor([SOS_token] * len(acronym), device=engine.device)
            dec_hidden = enc_hidden[0][-1]

            tf = random.random() < engine.teacher_forcing
            # Teacher forcing: Feed the target as the next input
            for di in range(ACRO_MAX_LENGTH):
                dec_output, dec_hidden, _ = engine.decoder._forward(dec_input, dec_hidden, enc_out)
                acro_loss += engine.morpheme_criterion(dec_output, acro_tensor[di])

                if tf:
                    dec_input = acro_tensor[di]  # Teacher forcing
                else:
                    topv, topi = dec_output.topk(1)
                    dec_input = topi.squeeze().detach()  # detach from history as input

            total_loss += acro_loss / acro_data.batch_size * engine.alpha
            acro_epoch_loss += acro_loss.item() / acro_data.batch_size

        total_loss.backward()
        engine.encoder_optim.step()
        engine.decoder_optim.step() 
        torch.cuda.empty_cache()

    return (prot_epoch_loss, morph_epoch_loss, acro_epoch_loss)

if __name__ == '__main__':
    args = docopt(__doc__)
    sid = hashlib.sha256(str(args).encode()).hexdigest()[-8:]
    args['id'] = sid
    print(args)

    # save the args of this run
    with open(f'files/{sid}.json', 'w') as w:
        w.write(json.dumps(args))

    epochs = int(args['--epochs'])
    bs = int(args['--batch_size'])
    best_vloss = 10.

    prot_train_data = dataset.ProteinData(os.path.join(args['--protein_data'], 'train.txt'), batch_size=bs, empty=args['--no_proteins'])
    prot_val_data   = dataset.ProteinData(os.path.join(args['--protein_data'], 'val.txt'), batch_size=bs, empty=args['--no_proteins'])
    acronym_data    = dataset.AcronymData(args['--acronym_data'], batch_size=bs, empty=args['--no_acronyms']) 
    morph_data      = dataset.MorphemeData(args['--morphology_data'], batch_size=bs, empty=args['--no_morphemes']) 

    args['--input_dim'] = VOCAB_SIZE + morph_data.num_tags

    engine = models.Engine(args)

    for e in range(epochs):
        loss = train(engine=engine, 
                     prot_data=prot_train_data, 
                     morph_data=morph_data,
                     acro_data=acronym_data)

        eval_results = evaluate.run(engine, 
                [('loss', prot_val_data), 
                 ('acc', morph_data),
                 ('auc', prot_val_data)]
                )
        vloss = eval_results[0]
        vacc = eval_results[1]
        vauc = eval_results[2]

        if vloss < best_vloss:
            torch.save(engine.encoder.state_dict(), f'files/{sid}.pkl')
            best_vloss = vloss

        print(f'Epoch {e} complete. Protein, Morph, Acron,\tValidation\tMorph Acc\tAUC')
        print(f'Epoch {e} complete. {loss[0]:.4f}, {loss[1]:.4f}, {loss[2]:.4f}\t{vloss:.4f}\t\t{vacc:.4f}\t\t{vauc:.4f}')

        with open('scoreboard.txt', 'a') as w:
            w.write(f"{sid},{vauc:.4f},{vloss:.4f},0.0,0.0\n")

