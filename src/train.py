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
import models
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

def train(engine: models.Engine, 
          prot_data: dataset.ProteinData, 
          morph_data: dataset.MorphemeData,
          acro_data: dataset.AcronymData,
          para_data: dataset.ParaData):

    engine.mSeq2Seq.train()
    prot_epoch_loss:float = 0
    morph_epoch_loss:float = 0
    acro_epoch_loss:float = 0
    para_epoch_loss:float = 0
    morph_data.shuffle()

    for idx in trange(max(len(prot_data), len(morph_data), len(acro_data), len(para_data))):
        if idx < len(morph_data):
            lemma, src = zip(*morph_data[idx])
            lemma_tensor = torch.stack(lemma).transpose(0,1).to(engine.device)

            engine.mOptimizer.zero_grad()
            output = engine.mSeq2Seq(src, lemma_tensor)
            mloss = F.nll_loss(output[1:].view(-1, VOCAB_SIZE), lemma_tensor[1:].contiguous().view(-1))
            mloss.backward()
            engine.mOptimizer.step()

            morph_epoch_loss += mloss.item()

        #torch.cuda.empty_cache()

    return (prot_epoch_loss, morph_epoch_loss, acro_epoch_loss, para_epoch_loss)

def fmt(flt):
    s = str(flt)
    if 'e' in s:
        s = '0.0'
    if len(s) <= 6:
        return s + ' '*(6-len(s))
    return s[:6]

def print_results(results, loss, tag):
    print(f"{tag} results:\t{fmt(loss)}\t{fmt(results['loss'])}\t{fmt(results['auc'])}\t{fmt(results['acc'])}")

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

    prot_train_data = dataset.ProteinData(os.path.join(args['--protein_data'], 'train.txt'), batch_size=bs, empty=args['--protein_data']=='')
    prot_val_data   = dataset.ProteinData(os.path.join(args['--protein_data'], 'val.txt'), batch_size=bs, empty=args['--protein_data']=='')
    acronym_data    = dataset.AcronymData(os.path.join(args['--acronym_data'], 'train.txt'), batch_size=bs, empty=args['--acronym_data']=='') 
    acronym_val_data = dataset.AcronymData(os.path.join(args['--acronym_data'], 'val.txt'), batch_size=bs, empty=args['--acronym_data']=='') 
    morph_data      = dataset.MorphemeData(os.path.join(args['--morpheme_data'], 'train.txt'), batch_size=bs, empty=args['--morpheme_data']=='') 
    morph_val_data  = dataset.MorphemeData(os.path.join(args['--morpheme_data'], 'val.txt'), batch_size=bs, empty=args['--morpheme_data']=='') 
    para_data       = dataset.ParaData(os.path.join(args['--paraphrase_data'], 'train.txt'), batch_size=bs, empty=args['--paraphrase_data']=='') 
    para_val_data   = dataset.ParaData(os.path.join(args['--paraphrase_data'], 'val.txt'), batch_size=bs, empty=args['--paraphrase_data']=='') 

    engine = models.Engine(args)

    for e in range(epochs):
        loss = train(engine=engine, 
                     prot_data=prot_train_data, 
                     morph_data=morph_data,
                     acro_data=acronym_data,
                     para_data=para_data)

        eval_dict = {'protein': {'data': prot_val_data, 'tests': ['loss', 'auc']}}
        results = evaluate.run(engine, eval_dict)

        if args['--protein_data']=='' or results['protein']['loss'] < best_vloss: 
            #torch.save(engine.encoder.state_dict(), f'files/{sid}.encoder.pkl')
            torch.save(engine.mSeq2Seq.state_dict(), f'files/{sid}.morpheme.pkl')
            #torch.save(engine.acro_decoder.state_dict(), f'files/{sid}.adecoder.pkl')
            #torch.save(engine.para_decoder.state_dict(), f'files/{sid}.adecoder.pkl')
            best_vloss = results['protein']['loss']

        if (e + 1) % int(args['--print_every']) == 0:
            eval_dict = {
                    'morpheme': {'data': morph_val_data, 'tests': ['acc']}
                    #'protein': {'data': prot_val_data, 'tests': ['loss', 'auc']},
                    #'paraphrase': {'data': para_val_data, 'tests': ['loss', 'acc']},
                    #'acronym': {'data': acronym_val_data, 'tests': ['loss', 'acc']}
                    }
            results = evaluate.run(engine, eval_dict)

            print(f'Epoch {e:02d} done.    \tTrain, \tValid.,\tV. AUC,\tV. Acc.')
            #print_results(results, loss=loss[0], tag='protein')
            print_results(results['morpheme'], loss=loss[1], tag='morpheme')
            #print_results(results, loss=loss[2], tag='acronym')
            #print_results(results, loss=loss[3], tag='paraphrase')

        #with open('scoreboard.txt', 'a') as w:
            #w.write(f"{shot}{sid},{results['protein']['auc']:.4f},{results['protein']['loss']:.4f},0.0,0.0\n")

