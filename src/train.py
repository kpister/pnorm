"""Train Decoder Architecture
Usage: 
    train.py [options]

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
    -p, --protein_data PATH             set the input file [default: ]
    -m, --morpheme_data PATH            set the morpheme input file [default: ]
    -a, --acronym_data PATH             set the acronym data file [default: ]
    -r, --paraphrase_data PATH          set the paraphrase data file [default: ]
    --print_every INT                   how often to print [default: 3]
"""

# personal libs
from constants import *
import evaluate
import engine as eng
import dataset

# other libs
import os
from typing import Dict, Any
import random
import torch 
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import trange #type: ignore
from docopt import docopt #type: ignore
import hashlib
import json

def invert(ls, ordering):
    output = torch.empty_like(ls)
    for i, ridx in enumerate(ordering):
        output[ridx] = ls[i]
    return output

def train(engine: eng.Engine, 
          prot_data: dataset.ProteinData, 
          morph_data: dataset.MorphemeData,
          acro_data: dataset.AcronymData,
          para_data: dataset.ParaData):

    if len(prot_data) > 0:
        engine.pEncoder.train()
        prot_data.shuffle()
    if len(morph_data) > 0:
        engine.mSeq2Seq.train()
        morph_data.shuffle()

    prot_epoch_loss:float = 0
    morph_epoch_loss:float = 0
    acro_epoch_loss:float = 0
    para_epoch_loss:float = 0

    for idx in trange(max(len(prot_data), len(morph_data), len(acro_data))):
        loss = torch.tensor([0.0]).to(engine.device)
        if idx < len(prot_data):
            x, x_lens, x_ord, y, y_lens, y_ord = prot_data[idx]

            engine.pOptimizer.zero_grad()
            hidden = engine.pEncoder.initHidden(x.size(0))
            x_enc, hid = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

            x_enc = invert(x_enc, x_ord)
            y_enc = invert(y_enc, y_ord)
            ploss = engine.pLoss._forward(x_enc, y_enc)

            loss += ploss
            prot_epoch_loss += ploss.item()

        if idx < len(morph_data):
            lemma, _, src, src_len = morph_data[idx]
            lemma = lemma.transpose(0,1).to(engine.device)
            src = src.to(engine.device)
            src_len = src_len.to(engine.device)

            engine.mOptimizer.zero_grad()
            output = engine.mSeq2Seq(src, src_len, lemma)
            mloss = F.nll_loss(output[1:].view(-1, VOCAB_SIZE), lemma[1:].contiguous().view(-1))

            loss += mloss
            morph_epoch_loss += mloss.item()

        loss.backward()
        if idx < len(prot_data):
            engine.pOptimizer.step()
        if idx < len(morph_data):
            engine.mOptimizer.step() 

    return (prot_epoch_loss, morph_epoch_loss, acro_epoch_loss, para_epoch_loss)

def fmt(flt):
    s = str(flt)
    if 'e' in s:
        s = '0.0'
    if len(s) <= 6:
        return s + ' '*(6-len(s))
    return s[:6]

def print_results(results, loss, tag):
    if loss == 0:
        out = f"{fmt(0)}\t{fmt(0)}\t{fmt(0)}\t{fmt(0)}"
    else:
        out = f"{fmt(loss)}\t{fmt(results['loss'])}\t{fmt(results['auc'])}\t{fmt(results['acc'])}"
    print(f"{tag} results:\t{out}")
    return out + "\t"

if __name__ == '__main__':
    args = docopt(__doc__)
    sid = hashlib.sha256(str(args).encode()).hexdigest()[-8:]
    args['id'] = sid
    print(args)

    # save the args of this run
    with open(f'files/{sid}.json', 'w') as w:
        w.write(json.dumps(args, indent=4, sort_keys=True))

    epochs = int(args['--epochs'])
    bs = int(args['--batch_size'])
    best_vloss = 10.

    shot = 'z' if 'zero_shot' in args['--protein_data'] else 'f'

    prot_data       = dataset.ProteinData(os.path.join(args['--protein_data'], 'train.txt'), batch_size=bs, empty=args['--protein_data']=='')
    prot_val_data   = dataset.ProteinData(os.path.join(args['--protein_data'], 'val.txt'), batch_size=bs, empty=args['--protein_data']=='')
    acro_data       = dataset.AcronymData(os.path.join(args['--acronym_data'], 'train.txt'), batch_size=bs, empty=args['--acronym_data']=='') 
    acro_val_data   = dataset.AcronymData(os.path.join(args['--acronym_data'], 'val.txt'), batch_size=bs, empty=args['--acronym_data']=='') 
    morph_data      = dataset.MorphemeData(os.path.join(args['--morpheme_data'], 'train.txt'), batch_size=bs, empty=args['--morpheme_data']=='') 
    morph_val_data  = dataset.MorphemeData(os.path.join(args['--morpheme_data'], 'val.txt'), batch_size=bs, empty=args['--morpheme_data']=='') 
    para_data       = dataset.ParaData(os.path.join(args['--paraphrase_data'], 'train.txt'), batch_size=bs, empty=args['--paraphrase_data']=='') 
    para_val_data   = dataset.ParaData(os.path.join(args['--paraphrase_data'], 'val.txt'), batch_size=bs, empty=args['--paraphrase_data']=='') 

    engine = eng.Engine(args)

    for e in range(epochs):
        loss = train(engine=engine, 
                     prot_data=prot_data, 
                     morph_data=morph_data,
                     acro_data=acro_data,
                     para_data=para_data)

        eval_dict = {'protein': {'data': prot_val_data, 'tests': ['loss', 'auc']}}
        results = evaluate.run(engine, eval_dict)

        # short circuit the or on empty proteins
        if args['--protein_data']=='' or results['protein']['loss'] < best_vloss: 
            if len(prot_data) > 0:
                torch.save(engine.pEncoder.state_dict(), f'files/{sid}.protein.pkl')
                best_vloss = results['protein']['loss']
            if len(morph_data) > 0:
                torch.save(engine.mSeq2Seq.state_dict(), f'files/{sid}.morpheme.pkl')
            if len(acro_data) > 0:
                torch.save(engine.aSeq2Seq.state_dict(), f'files/{sid}.acronym.pkl')
            if len(para_data) > 0:
                torch.save(engine.pSeq2Seq.state_dict(), f'files/{sid}.paraphrase.pkl')

        if (e + 1) % int(args['--print_every']) == 0:
            eval_dict = {}
            if len(morph_data) > 0:
                eval_dict['morpheme'] = {'data': morph_val_data, 'tests': ['acc']}
            if len(prot_data) > 0:
                eval_dict['protein'] = {'data': prot_val_data, 'tests': ['loss', 'auc']}
            if len(acro_data) > 0:
                eval_dict['acronym'] = {'data': acro_val_data, 'tests': ['acc']}
            if len(para_data) > 0:
                eval_dict['paraphrase'] = {'data': para_val_data, 'tests': ['acc']}
            results = evaluate.run(engine, eval_dict)

            print(f'Epoch {e:02d} done.    \tTrain, \tValid.,\tV. AUC,\tV. Acc.')
            output  = print_results(results['protein'], loss=loss[0], tag='protein')
            output += print_results(results['morpheme'], loss=loss[1], tag='morpheme')
            output += print_results(results['acronym'], loss=loss[2], tag='acronym')
            output += print_results(results['paraphrase'], loss=loss[3], tag='paraphrase')

            with open('scoreboard.txt', 'a') as w:
                w.write(f"{shot}{sid},{output}\n")
