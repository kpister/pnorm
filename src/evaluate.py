"""Evaluate
Usage:
    evaluate.py --config=FILE
"""

import engine as eng
import evaluate
import dataset

from torch.nn import functional as F
import random
from docopt import docopt
import sklearn.metrics as metrics #type: ignore
import torch
from typing import List, Tuple, Dict
from constants import *
import json
import os

spin = ['|', '/', '-', '\\']

def invert(ls, ordering):
    output = torch.empty_like(ls)
    for i, ridx in enumerate(ordering):
        output[ridx] = ls[i]
    return output

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0 # remove nan
    return X

def run(engine, cmds:Dict[str, Dict]) -> Dict[str, Dict]:
    ret = {}

    for k,v in cmds.items():
        ret[k] = {'loss': 0., 'acc': 0., 'auc': 0.}
        for test in v['tests']:
            if test == 'loss':
                ret[k]['loss'] = loss(engine, v['data'])
            if test == 'acc':
                ret[k]['acc'], _, ret[k]['loss'] = acc(engine, v['data'], k)
            if test == 'auc':
                ret[k]['auc'] = auc(engine, v['data'])
    print(' ' * 75, end='\r')
    return ret

def loss(engine, data):
    if len(data) == 0:
        return 0.

    engine.pEncoder.eval()
    loss = 0

    with torch.no_grad():
        for idx in range(len(data)):
            x, x_lens, x_ord, y, y_lens, y_ord = data[idx]
            hidden = engine.pEncoder.initHidden(x.size(0))
            x_enc, hid = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

            x_enc = invert(x_enc, x_ord)
            y_enc = invert(y_enc, y_ord)
 
            loss += engine.pLoss._forward(x_enc, y_enc).item()
            print(f'Evaluating {spin[idx%4]}', end='\r')

    return loss / len(data)

def tensor2word(t:torch.Tensor) -> str:
    return ''.join([chr(i) if i > 41 else ' ' for i in t])

def acc(engine, data, dtype):
    if len(data) == 0:
        return 0.,0., 0.

    if dtype == 'acronym':
        dec = engine.acro_decoder
        LENGTH = ACRO_MAX_LENGTH
    elif dtype == 'morpheme':
        model = engine.mSeq2Seq
        LENGTH = MAX_LENGTH
    elif dtype == 'paraphrase':
        dec = engine.para_decoder
        LENGTH = PARA_MAX_LENGTH

    model.eval()

    total = 0
    correct = 0
    distance = 0
    loss = 0

    with torch.no_grad():
        for idx in range(len(data)):
            trg, _ , src, src_len = data[idx]
            trg = trg.transpose(0,1).to(engine.device)
            src = src.to(engine.device)
            src_len = src_len.to(engine.device)

            output = engine.mSeq2Seq(src, src_len, trg)
            loss += F.nll_loss(output[1:].view(-1,VOCAB_SIZE), trg[1:].contiguous().view(-1))

            for true, gen in zip(trg.transpose(0,1), torch.argmax(output, 2).transpose(0,1)):
                #if random.random() < 0.001:
                    #print(f'{tensor2word(true)}:{tensor2word(gen)}')
                # This is currently strict equality
                # TODO consider equality up to first EOS_token
                if all(true == gen): 
                    correct += 1
                total += 1
                # TODO levenshtein distance
                #distance += leven(true, gen)

            print(f'Evaluating {spin[idx%4]}', end='\r')

    return correct / total, distance / total, loss.item()

def auc(engine, data):
    if len(data) == 0:
        return 0.

    resolution = 100
    min_distance = 0
    max_distance = 2
    true_pos_buckets = [0 for _ in range(resolution)]
    true_neg_buckets = [0 for _ in range(resolution)]
    increments = (max_distance - min_distance) / resolution
    total_tested = 0.

    engine.encoder.eval()
    with torch.no_grad():
        for idx in range(len(data)):
            x, x_lens, x_ord, y, y_lens, y_ord = data[idx]
            hidden = engine.pEncoder.initHidden(x.size(0))
            x_enc, hid = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

            x_enc = invert(x_enc, x_ord)
            y_enc = invert(y_enc, y_ord)

            #euclidean distance
            dist = torch.norm(y_enc.sub(x_enc), dim=1)
            #cosine similarity
            #dist = torch.nn.functional.cosine_similarity(l2norm(x_embedding), l2norm(y_embedding))


            for drop in range(resolution):
                true_pos_buckets[drop] += len(dist[dist<drop*increments])

            # Negative testing
            x, x_lens, x_ord, y, y_lens, y_ord = data.get_neg(x.size(0))

            hidden = engine.pEncoder.initHidden(x.size(0))
            x_enc, hid = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

            x_enc = invert(x_enc, x_ord)
            y_enc = invert(y_enc, y_ord)

            dist = torch.norm(y_enc.sub(x_enc), dim=1)

            for drop in range(resolution):
                true_neg_buckets[drop] += len(dist[dist>=drop*increments])

            total_tested += len(x)
            print(f'Evaluating {spin[idx%4]}', end='\r')

    false_pos_rate = [(total_tested - tn)/total_tested for tn in true_neg_buckets] +[1.]# fpr = _/_
    true_pos_rate = [tp/total_tested for tp in true_pos_buckets] + [1.]# tpr = _/_
    roc_auc = metrics.auc(false_pos_rate, true_pos_rate)

    # used to plot
    #with open('tmp_plot.txt', 'w') as e:
        #for x,y in zip(false_pos_rate,true_pos_rate):
            #e.write(f'{x},{y}\n')

    return roc_auc

if __name__ == '__main__':
    args = docopt(__doc__)

    # save the args of this run
    with open(args['--config']) as f:
        cfg = json.loads(f.read())

    bs = int(cfg['--batch_size'])
    bs = 1

    prot_val_data      = dataset.ProteinData(os.path.join(cfg['--protein_data'], 'val.txt'), batch_size=bs)
    morph_data         = dataset.MorphemeData(cfg['--morphology_data'], batch_size=bs) 
    cfg['--input_dim'] = VOCAB_SIZE + morph_data.num_tags

    engine = eng.Engine(cfg)
    eval_dict = {'protein' : {'data': prot_val_data, 'tests': ['loss', 'auc']}}
    eval_results = evaluate.run(engine, eval_dict)
    print(eval_results)
