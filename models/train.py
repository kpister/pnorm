'''Siamese Training
Usage:
    train.py --train=FILE --test=FILE [options]

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
'''

import torch
from docopt import docopt
import model as M
import dataset
import loss
import time
import torch.nn.functional as F
import math
import os


vocab_size = 128

def load_data(filename, quant):
    return dataset.ProteinData(filename, quant)

def load_model(args, device):
    model = M.Siamese(input_dim=vocab_size, hidden_dim=int(args['--LSTM_NODES']), device=device)
    if args['--LOAD_WEIGHTS']:
        model.load_state_dict(torch.load(args['--LOAD_WEIGHTS'], map_location=device))
    optim = torch.optim.Adam(model.parameters(), lr=float(args['--LEARNING_RATE']))
    criterion = loss.ContrastiveLoss(float(args['--MARGIN']))
    return (model, optim, criterion)

def train(model, data, batch_size, optim, criterion, device):
    model.train()
    epoch_loss= 0
    data.shuffle()

    for i in range(0, len(data) + batch_size, batch_size):

        end = min(i+batch_size, len(data))
        if end <= i: 
            break

        prot1, prot2 = data[i:end]
        out1, out2 = model(prot1, prot2)
        if len([i for i,v in enumerate(out1) if math.isnan(v[0])] + [i for i,v in enumerate(out2) if math.isnan(v[0])]) !=0:
            continue

        optim.zero_grad()
        loss = criterion(out1, out2)
        loss.backward()
        epoch_loss += loss.item()

        optim.step()
        print(f'Training progress: {i*100.0/len(data):.2f}%', end='\r')
    print(' '*50, end='\r')
    return epoch_loss / len(data) * batch_size

def evaluate(model, val_data, batch_size, device):
    loss = 0
    model.eval()

    with torch.no_grad():
        for i in range(0, len(val_data) + batch_size, batch_size):
            end = min(i+batch_size, len(val_data))
            if end <= i: 
                break

            prot1, prot2 = val_data[i:end]

            out1, out2 = model(prot1, prot2)
            if len([i for i,v in enumerate(out1) if math.isnan(v)] + [i for i,v in enumerate(out2) if math.isnan(v)]) !=0:
                continue
            loss += criterion(out1, out2).item()

    print(' '*50, end='\r')
    return (loss / len(val_data))

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

    train_data = load_data(args['--train'], -1)
    #val_data   = load_data(args['--val'], -1)
    test_data  = load_data(args['--test'], -1)

    # Create Model
    device = torch.device(args['--DEVICE'])
    model, optim, criterion = load_model(args, device)
    model.to(device)

    # track validation accuracy for early stopping
    highest_val = -1
    best_e = -1
    val = [0 for _ in range(6)] # stop when worse than the prev 4 epochs

    start = time.time()
    for e in range(epochs):
        loss = train(model, train_data, batch_size, optim, criterion, device)
        #vacc = evaluate(model, val_data, batch_size, device)
        #if vacc > highest_val:
            #highest_val, best_e = vacc, e
        vacc = e/100.

        # Print epoch, runtime, loss, and validation accuracy
        print(f'{e}\t{e*100//epochs}%\t({timeSince(start)})\tloss:{loss:.4f}')

        # Add validation testing for early stopping
        if vacc < min(val[:-2]) and min(val[:-2]) >= val[-1] and min(val[:-2]) >= val[-2]:
            print(f'Finished early on epoch {e}, best validation accuracy on epoch {best_e}.')
            break

        val = val[1:]
        val.append(vacc)

        # Save checkpoints
        if e % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args['--CHECKPOINT_DIR'],
                                                    f'model_{e}e_{int(vacc*1000)}v.pkl'))

    # evaluate on test
    acc = evaluate(model, test_data, batch_size, device)
    print(f'Accuracy on test set: {acc}')

    # Save final model. Note: this is not necessarily the best version
    # in fact it will often be the worst recent model
    torch.save(model.state_dict(), './siamese.pkl')
