"""Evaluate
Usage:
    evaluate.py --config=FILE.json
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
def pad_lists(lists, pad_int, device, pad_len=None, dtype=torch.float):
    """Pad lists (in a list) to make them of equal size and return a tensor."""

    if pad_len is None:
        pad_len = max([len(lst) for lst in lists])
    new_list = []
    for lst in lists:
        if len(lst) < pad_len:
            new_list.append(lst + [pad_int] * (pad_len - len(lst)))
        else:
            new_list.append(lst[:pad_len])
    return torch.tensor(new_list, dtype=dtype, device=device)

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
            x_enc, hid, _ = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid, _ = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

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
            lemma, lem_len, trg, _, tag = data[idx]
            #lemma = lemma.to(engine.device)
            #lem_len = lem_len.to(engine.device)
            #tag = tag.to(engine.device)
            #trg = trg.transpose(0,1).to(engine.device)

            p_ws, _, _ = model(lemma, tag)
            #p_ws, _, _ = model(lemma, lem_len, trg.size(0), tag)
            trg = pad_lists(trg, EOS_token, engine.device, pad_len=25, dtype=torch.long).transpose(0,1).to(engine.device)
            #trg = torch.stack(trg).transpose(0,1).to(engine.device)
            loss += F.nll_loss(p_ws.view(-1, VOCAB_SIZE), trg.contiguous().view(-1)) / len(lemma)

            for true, gen in zip(trg.transpose(0,1), torch.argmax(p_ws, 2)):
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

    engine.pEncoder.eval()
    with torch.no_grad():
        for idx in range(len(data)):
            x, x_lens, x_ord, y, y_lens, y_ord = data[idx]
            hidden = engine.pEncoder.initHidden(x.size(0))
            x_enc, hid, _ = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid, _ = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

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
            x_enc, hid, _ = engine.pEncoder(x.to(engine.device), x_lens, hidden, norm=True)
            y_enc, hid, _ = engine.pEncoder(y.to(engine.device), y_lens, hidden, norm=True)

            x_enc = invert(x_enc, x_ord)
            y_enc = invert(y_enc, y_ord)

            dist = torch.norm(y_enc.sub(x_enc), dim=1)

            for drop in range(resolution):
                true_neg_buckets[drop] += len(dist[dist>=drop*increments])

            total_tested += len(x)
            print(f'Evaluating {spin[idx%4]}', end='\r')

    false_pos_rate = [(total_tested - tn)/total_tested for tn in true_neg_buckets] +[1.]# fpr = fp/(fp+tn)
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
    #bs = 1

    prot_val_data      = dataset.ProteinData(os.path.join(cfg['--protein_data'], 'val.txt'), batch_size=bs)
    morph_data         = dataset.MorphemeData(os.path.join(cfg['--morpheme_data'], 'val.txt'), batch_size=bs) 
    cfg['--tag_size'] = morph_data.num_tags

    engine = eng.Engine(cfg)
    eval_dict = {'protein' : {'data': prot_val_data, 'tests': ['loss', 'auc']}}
    eval_results = evaluate.run(engine, eval_dict)
    print(eval_results)
