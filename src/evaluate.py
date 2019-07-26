"""Evaluate
Usage:
    evaluate.py --config=FILE
"""

import models
import evaluate
import dataset

import random
from docopt import docopt
import sklearn.metrics as metrics #type: ignore
import torch
from typing import List, Tuple
from constants import *
import json
import os

spin = ['|', '/', '-', '\\']

def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0 # remove nan
    return X

def run(engine, cmds:List[Tuple]) -> List[float]:
    ret = []

    for k,v in cmds:
        if k == 'loss':
            ret.append(loss(engine, v))
        if k == 'acc':
            racc, rdist = acc(engine, v)
            ret.append(racc)
        if k == 'auc':
            ret.append(auc(engine, v))
        print('Evaluating  \t' + '\t'.join([f'{r:.4f}' for r in ret]), end='\r')
    return ret

def loss(engine, data):
    if len(data) == 0:
        return 0

    engine.encoder.eval()
    loss = 0

    with torch.no_grad():
        for idx in range(len(data)):
            x, y = zip(*data[idx])
            enc_hidden = engine.encoder.initHidden(batch_size=len(x))

            x_embedding, _hs = engine.encoder._forward(x, enc_hidden)
            y_embedding, _hs = engine.encoder._forward(y, enc_hidden)
 
            loss += engine.protein_criterion._forward(x_embedding, y_embedding).item()
            print(f'Evaluating {spin[idx%4]}', end='\r')

    return loss / len(data)

def tensor2word(t:torch.Tensor) -> str:
    return ''.join([chr(i) if i > 41 else ' ' for i in t])

def acc(engine, data):
    if len(data) == 0:
        return 0

    engine.encoder.eval()
    engine.decoder.eval()

    total = 0
    correct = 0
    distance = 0

    with torch.no_grad():
        for idx in range(len(data)):
            lemma, input_tensor = zip(*data[idx])
            lemma_tensor = torch.stack(lemma).transpose(0,1).to(engine.device)

            enc_hidden = engine.encoder.initHidden(batch_size=len(input_tensor))
            enc_out, enc_hidden = engine.encoder._forward_sorted( #type:ignore
                    input_tensor, hidden=enc_hidden)

            # Decode morpheme
            dec_input = torch.tensor([SOS_token] * len(lemma), device=engine.device)
            dec_hidden = enc_hidden[0][-1]

            word_builder = torch.zeros((MAX_LENGTH, len(lemma)), device=engine.device)
            # use its own predictions as the next input
            for di in range(MAX_LENGTH):
                dec_output, dec_hidden, _ = engine.decoder._forward(
                    dec_input, dec_hidden, enc_out)
                topv, topi = dec_output.topk(1)
                dec_input = topi.squeeze().detach()  # detach from history as input
                word_builder[di] = dec_input

            word_builder = word_builder.transpose(0,1).long()
            lemma_tensor = lemma_tensor.transpose(0,1)
            for true, gen in zip(lemma_tensor, word_builder):
                if random.random() < 0.001:
                    print(f'{tensor2word(true)}:{tensor2word(gen)}')
                # This is currently strict equality
                # TODO consider equality up to first EOS_token
                if all(true == gen): 
                    correct += 1
                total += 1
                # TODO levenshtein distance
                #distance += leven(true, gen)

            print(f'Evaluating {spin[idx%4]}', end='\r')

    return correct / total, distance / total

def auc(engine, data):
    if len(data) == 0:
        return 0

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
            x, y = zip(*data[idx])

            enc_hidden = engine.encoder.initHidden(batch_size=len(x))
            x_embedding, _hs = engine.encoder._forward(x, enc_hidden)
            y_embedding, _hs = engine.encoder._forward(y, enc_hidden)

            #euclidean distance
            dist = torch.norm(y_embedding.sub(x_embedding), dim=1)
            #cosine similarity
            #dist = torch.nn.functional.cosine_similarity(l2norm(x_embedding), l2norm(y_embedding))


            for drop in range(resolution):
                true_pos_buckets[drop] += len(dist[dist<drop*increments])

            # Negative testing
            x, y = data.get_neg(len(x))
            x_embedding, _hs = engine.encoder._forward(x, enc_hidden)
            y_embedding, _hs = engine.encoder._forward(y, enc_hidden)

            dist = torch.norm(y_embedding.sub(x_embedding), dim=1)

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

    engine = models.Engine(cfg)
    eval_results = evaluate.run(engine, [ ('auc', prot_val_data)])
    print(eval_results)
