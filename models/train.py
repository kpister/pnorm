'''Siamese Training
Usage:
    train.py --input=DIR --minput=FILE [options]

Options:
    -h, --help              show this message and exit
    --ACTIVATION NAME       set the activation type (relu, tanh) [default: relu]
    --BATCH_SIZE SIZE       set BATCH_SIZE [default: 512]
    --CHECKPOINT_DIR BASE   set the checkpoint directory [default: ./checkpoints/]
    --DEVICE DEV            set the device (cuda:0, cuda:1, cpu) [default: cuda:1]
    --DROPOUT FLOAT         set the dropout rate [default: 0.3]
    --CHAR_EMBEDDING NUM    set the size of the char embedded space [default: 100]
    --ENCODER TYPE          set the type of encoder (lstm, cnn, gru) [default: lstm]
    --EPOCHS COUNT          set max number of epochs [default: 40]
    --LAYERS NUM            set the number of layers in the encoder [default: 5]
    --LEARNING_RATE FLOAT   set learning rate for adam optimizer [default: 0.001]
    --LOAD_WEIGHTS FILE     set old weights to learn from
    --HIDDEN NUM            set the number of nodes in the hidden layer [default: 120]
    --MARGIN FLOAT          set the loss cutoff margin [default: 2.0]
    --SILENT                set if printing should happen [default: False]
    --WORD_EMBEDDING NUM    set the size of the word embedded space [default: 100]
'''

import torch
from docopt import docopt
import model as M
import dataset
import loss
import time
import json
import math 
import os

vocab_size = 128

def load_model(args, device):
    opts = {}
    opts['char_embedding'] = int(args['--CHAR_EMBEDDING'])
    opts['word_embedding'] = int(args['--WORD_EMBEDDING'])
    opts['bidirectional'] = True
    opts['activation'] = args['--ACTIVATION']
    opts['input_dim'] = vocab_size
    opts['dropout'] = float(args['--DROPOUT'])
    opts['encoder'] = args['--ENCODER']
    opts['layers'] = int(args['--LAYERS'])
    opts['hidden'] = int(args['--HIDDEN'])
    opts['device'] = device

    model = M.Siamese(opts)

    if args['--LOAD_WEIGHTS']:
        model.load_state_dict(torch.load(args['--LOAD_WEIGHTS'], map_location=device))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=float(args['--LEARNING_RATE']))
    criterion = loss.ContrastiveLoss(float(args['--MARGIN']))

    return {'model':model, 'optim':optim, 'loss':criterion}

def train(engine, data, batch_size, device, silent):
    engine['model'].train()
    epoch_loss= 0
    data.shuffle()

    for i in range(0, len(data) + batch_size, batch_size):
        end = min(i+batch_size, len(data))
        if end <= i: break

        prot1, prot2 = data[i:end]
        out1, out2 = engine['model'](prot1, prot2)

        engine['optim'].zero_grad()
        loss = engine['loss'](out1, out2)
        loss.backward()
        epoch_loss += loss.item()

        engine['optim'].step()
        if not silent:
            print(f'Training progress: {i*100.0/len(data):.2f}%', end='\r')

    if not silent:
        print(' '*50, end='\r')
    return epoch_loss / len(data) * batch_size

def evaluate(engine, data, batch_size, device, silent):
    engine['model'].eval()
    loss = 0

    with torch.no_grad():
        for i in range(0, len(data) + batch_size, batch_size):
            end = min(i+batch_size, len(data))
            if end <= i: break

            prot1, prot2 = data[i:end]
            loss += engine['loss'](*(engine['model'](prot1, prot2))).item()
            if not silent: 
                print(f'Eval progress: {i*100.0/len(data):.2f}%', end='\r')

    if not silent:
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
    if not os.path.isdir(args['--CHECKPOINT_DIR']):
        os.mkdir(args['--CHECKPOINT_DIR'])
    with open(os.path.join(args['--CHECKPOINT_DIR'], 'config.json'), 'w') as c:
        c.write(json.dumps(args, indent=2))

    print(args)

    epochs     = int(args['--EPOCHS'])
    batch_size = int(args['--BATCH_SIZE'])

    pos = [os.path.join(args['--input'], 'train.txt'), args['--minput']]
    print(pos)
    train_data = dataset.ProteinData(pos)
    val_data = dataset.ProteinData(os.path.join(args['--input'], 'val.txt'))

    # Create Model
    device = torch.device(args['--DEVICE'])
    engine = load_model(args, device)

    # track validation accuracy for early stopping
    best_vloss = -1
    best_epoch = -1
    best_model = None

    start = time.time()
    for e in range(epochs):
        loss  = train(engine, train_data, batch_size, device, args['--SILENT'])
        vloss = evaluate(engine, val_data, batch_size, device, args['--SILENT'])

        # Print epoch, runtime, loss, and validation loss
        print(f'{e}\t{e*100//epochs}%\t({timeSince(start)})\tloss:{loss:.4f}\tval loss:{vloss:.4f}')

        # Save checkpoints
        if vloss < best_vloss or best_vloss == -1:
            torch.save(engine['model'].state_dict(), os.path.join(args['--CHECKPOINT_DIR'], 'best_model.pkl'))
            best_vloss = vloss
            best_model = engine['model']
            best_epoch = e
        elif e - best_epoch >= 5:
            # reset the weights for maybe a better approach
            engine['model'] = best_model
            engine['optim'] = torch.optim.Adam(engine['model'].parameters(), lr=(float(args['--LEARNING_RATE'])*0.5*(e//10)))
            best_epoch = e
            print('Reseting to best found weights')
            # try a different loss? try a different optimizer? change learning rate?

    # evaluate on test
    engine['model'] = best_model
    floss = evaluate(engine, val_data, batch_size, device, args['--SILENT'])
    print(f'Final accuracy on val set: {floss:.4f}')
    with open('scoreboard.txt', 'a') as w:
        w.write(f"{floss:.4f},{args['--LSTM_NODES']},{args['--LAYERS']},{args['--EMBEDDING_DIM']},{best_epoch}\n")
