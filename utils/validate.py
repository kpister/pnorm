"""Validate Word
Usage:
    validate.py (<word> | --file=<in-file>) [options]

Options:
    -h, --help          show this message
    -m, --model FILE    set model file [default: ./models/model.pkl]
    -d, --device DEV    set the device {'cuda', 'cpu'} [default: cuda]
    -p, --path PATH     set path to the model file [default: ./models/]
"""

import torch
import torch.nn.functional as F

def predict_ordered_batch(x, model, vocab_size=128, batch_size=1000):
    if len(x) == 0:
        return []

    x = [i for i in x if len(i) < 100]

    predictions = []
    with torch.no_grad():
        for i in range(0, len(x) + batch_size, batch_size):
            end = min(i+batch_size, len(x))
            if end <= i: 
                break

            batch = sorted(x[i:end], key=lambda v:len(v[0]), reverse=True)
            predictions += [z for z in zip(batch, model.ordered_forward(batch))]

    return predictions


def predict_batch(x, model, vocab_size=128, batch_size=1000):
    if len(x) == 0:
        return []

    x = [i for i in x if len(i) < 100]

    predictions = []
    with torch.no_grad():
        for i in range(0, len(x) + batch_size, batch_size):
            end = min(i+batch_size, len(x))
            if end <= i: 
                break

            batch = sorted(x[i:end], key=lambda v:len(v), reverse=True)
            predictions += [z for z in zip(batch, model(batch))]

    return predictions

def get_siamese_model(filename, device='cpu'):
    import sys
    sys.path.append('./models')
    from siamese_model import Siamese

    vocab_size = 128
    hidden_dim = 120
    device = torch.device(device)
    model = Siamese(vocab_size, hidden_dim, device)
    model.eval()
    return model

def predict_siamese(model):
    return lambda w1, w2: model.batchless(w1, w2)

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    import sys
    sys.path.append(args['--path'])
    from classifier import LSTMClassifier

    device = torch.device(args['--device'])
    model = LSTMClassifier(128, 120, 3, device)
    model.load_state_dict(torch.load(args['--model'], map_location=device))
    model.to(device)
    model.eval()
    
    if args['--file']:
        with open(args['--file']) as f:
            targets = f.read().split('\n')
        targets = [t for t in targets if len(t) > 0]
    else:
        targets = [args['<word>']]

    for p in predict_batch(targets, model):
        print(f'{p[0]}: {p[1][0]:.2f}, {p[1][1]:.2f}, {p[1][2]:.2f}')
