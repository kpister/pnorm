"""Siamese Baseline
Usage:
    eval.py --input=FILE --output=FILE --model=FILE
"""


from docopt import docopt
import dataset
from leven import levenshtein
import torch
import sys
sys.path.append('./models')
from model import Siamese

def eval(val_data, model):
    count = 100
    min_thresh = 0
    max_thresh = 1
    correct = [0,0]
    total = [0,0]
    batch_size = 1000
    best = 0
    thresh=0.6

    for i in range(0, len(val_data) + batch_size, batch_size):
        end = min(i+batch_size, len(val_data))
        if end <= i: 
            break

        prot1, prot2 = val_data[i:end]
        o1, o2 = model(prot1, prot2)
        res = torch.nn.functional.cosine_similarity(o1, o2, dim=1)

        for j in range(len(prot1)):
            if res[j] > thresh:
                correct[0] += 1
            else:
                print(res[j])
                print(asdf(prot2[j]))
                print(asdf(prot1[j]))
        total[0] += len(prot1)

        neg1, neg2 = val_data.get_neg(batch_size)
        n1, n2 = model(neg1, neg2)
        res = torch.nn.functional.cosine_similarity(n1, n2, dim=1)
        for j in range(len(neg1)):
            if res[j] <= thresh:
                correct[1] += 1
            else:
                print(f'neg {res[j]}')
                print(asdf(neg1[j]))
                print(asdf(neg2[j]))

        total[1] += len(neg1)

        s = ''
        for k in range(len(correct)):
            if k % 10 == 0:
                s = s + f'\t({correct[k][0][0]/total[0]*100.0:.1f},{correct[k][0][1]/total[1]*100.0:.1f})'
        print(f'Progress: {i*100.0/len(val_data):.1f}%{s}', end='\r')
        for lv in [((q[0][0] + q[0][1]) / (total[0] + total[1]), q[1]) for q in correct]:
            if lv[0] > best:
                best = lv[0]
                print(f'\nUpdated best value: {best}')

    print(' '*50, end='\r')
    return [(i, total) for i in correct]

def get_siamese_model(filename, device='cpu'):
    vocab_size = 128
    hidden_dim = 100
    device = torch.device(device)
    model = Siamese(vocab_size, hidden_dim, device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    val_data  = dataset.ProteinData(args['--input'], -1)
    model = get_siamese_model(args['--model'])
    vacc = evaluate_siamese(val_data, model)

    best = 0
    with open(args['--output'], 'w') as w:
        for v,tot in vacc:
            x = (v[0][0] + v[0][1])/(tot[0]+tot[1])
            if x > best:
                best = x
                print(f'Best threshold: {v[1]}, {best}')
            w.write(f'{v[1]:.3f},{v[0][0]/tot[0]},{v[0][1]/tot[1]},{(v[0][0]+v[0][1])/(tot[0]+tot[1])}\n')
        
