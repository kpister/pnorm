'''Siamese Training
Usage:
    train.py --train=FILE... --test=FILE [options]

Options:
    -h, --help              show this message and exit
    --test FILE             set the path of the test data file
    --train FILE            set the path of the training data file
    --val FILE              set the path of the validation data file
    --BATCH_SIZE SIZE       set BATCH_SIZE [default: 64]
    --CHECKPOINT_DIR BASE   set the checkpoint directory [default: ./checkpoints/]
    --DEVICE DEV            set the device (cuda:0, cuda:1, cpu) [default: cuda:1]
    --EPOCHS COUNT          set max number of epochs [default: 20]
    --LEARNING_RATE FLOAT   set learning rate for adam optimizer [default: 0.001]
    --LSTM_NODES NUM        set the number of nodes in the hidden layer [default: 120]
    --MARGIN FLOAT          set the loss cutoff margin [default: 2.0]
    --LOAD_WEIGHTS FILE     set old weights to learn from
    --LAYERS NUM            set the number of layers in the encoder [default: 5]
    --EMBEDDING_DIM NUM     set the size of the embedded space [default: 100]
    --CE                    set CE [default: False]
'''

import torch
from docopt import docopt
import model as M
import dataset
import loss
import time
import math 
import os


vocab_size = 128

def load_data(filename):
    return dataset.ProteinData(filename)

def load_model(args, device):
    lstm_in = vocab_size if not args['--CE'] else 100

    model = M.Siamese(input_dim=lstm_in, 
                      hidden_dim=int(args['--LSTM_NODES']), 
                      output_dim=int(args['--EMBEDDING_DIM']),
                      num_layers=int(args['--LAYERS']),
                      device=device)

    if args['--LOAD_WEIGHTS']:
        model.load_state_dict(torch.load(args['--LOAD_WEIGHTS'], map_location=device))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=float(args['--LEARNING_RATE']))
    criterion = loss.ContrastiveLoss(float(args['--MARGIN']))

    cembed = M.CharEncoder(input_dim=vocab_size, output_dim=100, num_layers=2, device=device)
    char_optim = torch.optim.SGD(cembed.parameters(), lr=float(args['--LEARNING_RATE']))
    cembed.to(device)

    return {'model':model, 'cmodel':cembed, 'optim':optim, 'coptim':char_optim, 'loss':criterion, 'cembed':args['--CE']}

def train(engine, data, batch_size, device):
    engine['model'].train()
    engine['cmodel'].train()
    epoch_loss= 0
    data.shuffle()

    for i in range(0, len(data) + batch_size, batch_size):
        end = min(i+batch_size, len(data))
        if end <= i: break

        prot1, prot2 = data[i:end]
        # perform a character embedding
        if engine['cembed']:
            for idp in range(len(prot1)):
                prot1[idp] = engine['cmodel'](prot1[idp])
            for idp in range(len(prot2)):
                prot2[idp] = engine['cmodel'](prot2[idp])
            engine['coptim'].zero_grad()

        out1, out2 = engine['model'](prot1, prot2)

        engine['optim'].zero_grad()
        loss = engine['loss'](out1, out2)
        loss.backward()
        epoch_loss += loss.item()

        engine['optim'].step()
        if engine['cembed']:
            engine['coptim'].step()
        print(f'Training progress: {i*100.0/len(data):.2f}%', end='\r')
    print(' '*50, end='\r')
    return epoch_loss / len(data) * batch_size

def evaluate(engine, data, batch_size, device):
    engine['model'].eval()
    engine['cmodel'].eval()
    loss = 0

    with torch.no_grad():
        for i in range(0, len(data) + batch_size, batch_size):
            end = min(i+batch_size, len(data))
            if end <= i: break

            prot1, prot2 = data[i:end]
            if engine['cembed']:
                for idp in range(len(prot1)):
                    prot1[idp] = engine['cmodel'](prot1[idp])
                for idp in range(len(prot2)):
                    prot2[idp] = engine['cmodel'](prot2[idp])

            loss += engine['loss'](*(engine['model'](prot1, prot2))).item()
            print(f'Eval progress: {i*100.0/len(data):.2f}%', end='\r')

    print(' '*50, end='\r')
    return (loss / len(data) * batch_size)

# Format time printing
def timeSince(since):
    now = time.time()
    s = int(now - since)
    m, s = s // 60, s % 60
    h, m = m // 60, m % 60
    return f'{h:02d}h {m:02d}m {s:02d}s'

if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    epochs     = int(args['--EPOCHS'])
    batch_size = int(args['--BATCH_SIZE'])

    train_data = load_data(args['--train'])
    test_data  = load_data(args['--test'])
    if args['--val']:
        val_data = load_data(args['--val'])
    else:
        val_data = test_data

    # Create Model
    device = torch.device(args['--DEVICE'])
    engine = load_model(args, device)

    # track validation accuracy for early stopping
    best_vloss = -1
    best_model = None

    start = time.time()
    for e in range(epochs):
        loss  = train(engine, train_data, batch_size, device)
        vloss = evaluate(engine, val_data, batch_size, device)

        # Print epoch, runtime, loss, and validation loss
        print(f'{e}\t{e*100//epochs}%\t({timeSince(start)})\tloss:{loss:.4f}\tval loss:{vloss:.4f}')

        # Save checkpoints
        if vloss < best_vloss or best_vloss == -1:
            torch.save(engine['model'].state_dict(), os.path.join(args['--CHECKPOINT_DIR'], 'best_model.pkl'))
            best_vloss = vloss
            best_model = engine['model']

    # evaluate on test
    tloss = evaluate(best_model, test_data, batch_size, criterion, device)
    print(f'Accuracy on test set: {tloss:.4f}')
