"""Siamese Baseline 
Usage:
    eval.py --input=FILE --output=FILE --model=FILE [options]

Options:
    --DEVICE STR        set the runner [default: cpu]
    --DISTANCE MTD      set the distance method {cos, euc} [default: euc]
"""


from docopt import docopt #type: ignore
import dataset
from leven import levenshtein #type: ignore
import torch
import sklearn.metrics as metrics #type: ignore
import sys
sys.path.append('./models')
from model import Siamese

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    X[X != X] = 0 # remove nan
    return X

def cos(x1, x2):
    return torch.nn.functional.cosine_similarity(l2norm(x1), l2norm(x2))
def euc(x1, x2):
    return torch.norm(x2.sub(x1), dim=1)

def classify(val_data, model, dist_metric):
    classes = [] # :: class
    unknown_data = [] # :: y, x :: true_class, mention
    correct = 0
    total = 0
    for y, x in unknown_data:
        y_h = (None, -1)
        u_embedding = model.embed(x)
        for c in classes:
            c_embedding = model.embed(c)
            dist = dist_metric(u_embedding, c_embedding)
            if dist < y_h[1]:
                y_h[1] = dist
                y_h[0] = c
        correct += y_h[0] == y
        total += 1
    return correct/total

def cos_c(x, thresh):
    return x > thresh
def euc_c(x, thresh):
    return x < thresh
    
def evaluate(val_data, model, dist_metric, correctness):
    count = 100
    min_thresh = 0
    max_thresh = 3
    correct = [[0,0] for i in range(count)]
    total = [0,0]
    batch_size = 1000
    best = 0

    for i in range(0, len(val_data) + batch_size, batch_size):
        end = min(i+batch_size, len(val_data))
        if end <= i: 
            break

        prot1, prot2 = val_data[i:end]
        o1, o2 = model(prot1, prot2)
        res = dist_metric(o1, o2)

        for j in range(len(prot1)):
            for k in range(count):
                if correctness(res[j], min_thresh + k / count * (max_thresh-min_thresh)):
                    correct[k][0] += 1
        total[0] += len(prot1)

        neg1, neg2 = val_data.get_neg(batch_size)
        n1, n2 = model(neg1, neg2)
        res = dist_metric(n1, n2)
        for j in range(len(neg1)):
            for k in range(count):
                if not correctness(res[j], min_thresh + k / count * (max_thresh-min_thresh)):
                    correct[k][1] += 1
        total[1] += len(neg1)

        s = ''
        for k in range(0, len(correct), len(correct)//10):
            s = s + f'\t({correct[k][0]/total[0]*100.0:.1f},{correct[k][1]/total[1]*100.0:.1f})'
        print(f'Progress: {i*100.0/len(val_data):.1f}%{s}', end='\r')
        for lv in [((q[0] + q[1]) / (total[0] + total[1]), q[1]) for q in correct]:
            if lv[0] > best:
                best = lv[0]
                print(f'\nUpdated best value: {best}')

    print(' '*50, end='\r')

    tpr = [(i[0]/total[0]) for i in correct]
    fpr = [(total[1] - i[1])/total[1] for i in correct]
    roc_auc = metrics.auc(fpr, tpr)

    with open('out.txt', 'w') as e:
        e.write(f'{roc_auc}\n')
        for x,y in zip(fpr,tpr):
            e.write(f'{x},{y}\n')

    return [(i, total) for i in correct]

def get_siamese_model(filename, device='cuda:0'):
    vocab_size = 128
    c_embedding_dim = 100
    hidden_dim = 300
    output_dim = 100
    num_layers = 5
    device = torch.device(device)
    model = Siamese(vocab_size, c_embedding_dim, hidden_dim, output_dim, 0.0, num_layers, device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    val_data  = dataset.ProteinData(args['--input'])
    model = get_siamese_model(args['--model'], args['--DEVICE'])

    d = euc if args['--DISTANCE'].lower() == 'euc' else cos
    dc = euc_c if args['--DISTANCE'].lower() == 'euc' else cos_c
    vacc = evaluate(val_data, model, dist_metric=d, correctness=dc)

    best = 0
    with open(args['--output'], 'w') as w:
        for v,tot in vacc:
            x = (v[0] + v[1])/(tot[0]+tot[1])
            if x > best:
                best = x
                print(f'Best threshold: {v[1]}, {best}')
            w.write(f'{v[1]:.3f},{v[0]/tot[0]},{v[1]/tot[1]},{(v[0]+v[1])/(tot[0]+tot[1])}\n')
        
