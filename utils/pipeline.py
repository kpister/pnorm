"""Pipeline
Usage:
    pipeline.py <patent_file> [options]

Options: 
    -h, --help          show this message and exit 
    -b, --batch SIZE    set the batch size [default: 1000]
    -d, --device DEV    set the device to predict on {cuda:0, cuda:1, cpu} [default: cuda:0]
    -m, --model FILE    set model directory [default: ./models/model.pkl]
    -n, --ngrams SIZE   set the max n-gram length [default: 3]
"""

# personal tooling
from validate import predict_batch
from parse_patent import XMLDoc
from parse_tsv import PatentTSV
from preprocess import remove_common
from preprocess import preprocess
from choose import best_options
import check

class bcolors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

# Print out up to 10 of the best proteins from dic
def printProteins(name, dic):
    output = []
    for k,v in sorted(dic.items(), key=lambda kv:kv[1][1], reverse=True)[:10]:
        if v[1] > 0.1:
            output.append(f"{k} {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}")

    if len(output) == 0:
        print(f'{bcolors.FAIL}{name}:No protein matches{bcolors.ENDC}')
        return []

    prepend = [f'Found {len(dic)} possible targets from {name}',
               f'Printing top {len(output)}:',
               f"{'~'*50} prot, comp, norm"]
    return prepend + output

def predict(text, model, batch_size=500, ngram_size=3):
    ## Predict the text
    # ngrams  :: list[sequences]
    # model :: pytorch model

    preds = {}
    common = open('utils/words.txt').read().split('\n')
    ngrams = remove_common(text, common, ngram_size=ngram_size)
    #ngrams = preprocess(text, ngram_size=ngram_size)

    for pred in predict_batch(ngrams, model, batch_size=batch_size):
        preds[pred[0]] = pred[1] # stored in normal, protein, compound order

    return preds

def pipeline(patent_file, model, batch_size, ngram_size): 
    output = []

    ## Parse input xml file
    # filename should be direct path from current location
    # set intro to true when searching cits and intro
    doc = XMLDoc(patent_file)
    print(f'{bcolors.OKBLUE}Loaded {patent_file}{bcolors.ENDC}')

    ## Parse related tsv file for true targets
    # tsv file should be correlated in name
    # and located in the same directory
    try:
        tsvname = patent_file.replace("US0", "US")
        tsvname = tsvname[:tsvname.find("-")] + ".tsv"
        true_targets = '\n# '.join(PatentTSV(tsvname).targets)

        output += [f"True proteins (from tsv):\n# {true_targets}"]
    except Exception as e:
        print(f'{patent_file}:tsv:{e}\nContinuing')

    content = f'{doc.title}\n{doc.abstract}\n{doc.whole_invention}\n{doc.whole_patent}\n{doc.references}'
    #content = doc.whole
    preds = predict(content, model, batch_size, ngram_size)

    # stop to do clustering later
    return output + [f'{k} ~ {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}' for k, v in preds.items()]

    seqs  = [(k, v[1]) for k, v in preds.items() if v[1] > 0.5]

    if len(seqs) == 0:
        print("No sequences found")
        return []

    # check for membership in uniprot
    in_uniprot = check.check_batch([x[0] for x in seqs], check.get_master())
    for i in range(len(seqs)):
        seq = seqs[i]
        if len(in_uniprot[i]) > 0:
            output += [f'Found protein: {seq[0]} with names {in_uniprot[i]}, predicted {seq[1]}']

    clusters = best_options(seqs,distance=5)
    #TODO fix formatting
    if len(clusters) > 0:
        output += [f"Final Guess:\n& {clusters[0][0][0][0]}\n\n"]

    output += ['Guesses']

    for index, k in enumerate(clusters):
        cluster = k[0]
        output += [f'\nCluster {index}:\tAverage:{k[1]:.2f}']

        for guess in cluster: 
            output += [f'{guess[0]}, {guess[1].item():.3f}']

    output += ["\nDEBUGGING INFO", f'Title: {doc.title}', '\nALL PREDICTIONS (normal, prot, comp)\n']
    # add entire output for parsing later if needed
    output += [f'{k}, {v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}' for k, v in preds.items()]

    return output

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    import torch
    import sys
    sys.path.append('./models/')
    from classifier import LSTMClassifier

    device = torch.device(args['--device'])
    model = LSTMClassifier(128, 120, 3, device)
    model.load_state_dict(torch.load(args['--model'], map_location=device))
    model.to(device)
    model.eval()

    batch_size = int(args['--batch'])
    ngram_size = int(args['--ngrams'])

    output = pipeline(args['<patent_file>'], model, batch_size, ngram_size)
    print('\n'.join(output))
