"""CMD
Usage:
    cmd.py blend <pos-file> <neg-file> <out-file>
    cmd.py case <in-file> <out-file>
    cmd.py clean <in-file> <out-file>
    cmd.py cluster <info-dir> <model.pkl> <out-file> <device>
    cmd.py count_ngrams <in-file>
    cmd.py dedup <in-file> <out-file> <delimiter>
    cmd.py lower <in-file> <out-file> <delimiter>
    cmd.py parse_all_patents <patents-dir> <out-file>
    cmd.py parse_patent <xml-file> [--all --title --abstract --references --whole]
    cmd.py parse_all_tsvs <patents-dir> <out-file>
    cmd.py parse_tsv (--tsv=<tsv-file> | --xml=<xml-file>)
    cmd.py permute <in-file> <out-file>
    cmd.py split <in-file>
    cmd.py stem <in-file> <out-file>

Notes:
    blend: blend a two files randomly
    clean: Convert unicode to nearest ascii value in file, save to out-file
    cluster: given directory of .info files, output a single csv of top 2 clusters
    countngrams: Count the number of each length of n-gram in a file
    parse_all_patents: save all the normal content of the patent directory
    parse_patent: print out patent attributes
    parse_all_tsvs: output all tsv targets to file
    parse_tsv: read the targets of a tsv
    split: split input file into 60% training, 20% validation, 20% test
"""

from docopt import docopt
args = docopt(__doc__)

# counts number of each ngram
if args['count_ngrams']:
    with open(args['<in-file>']) as f:
        buckets = [0 for _ in range(10)]
        for line in f:
            ns = len(line.split(' '))
            if ns >= len(buckets):
                continue
            buckets[ns] += 1

    for i in range(1, 10):
        print(f'{i}-grams: {buckets[i]}')

# removes non-ascii values
if args['clean']:
    from unidecode import unidecode
    c = 0
    with open(args['<in-file>']) as f, open(args['<out-file>'], 'w') as w:
        lines = f.read().split('\n')
        for line in lines:
            data = unidecode(line.strip())
            w.write(data + '\n')
            c += data != line.strip()
        print(f'Made changes on {c} lines')

# blend two file inputs
if args['blend']:
    import random

    pos = [i.strip() for i in open(args['<pos-file>']).read().split('\n')]
    neg = [i.strip() for i in open(args['<neg-file>']).read().split('\n')]
    neg = [i for i in neg if len(i.split(' ')) < 3]

    quant = min([5000000, len(neg), len(pos)])
    w = open(args['<out-file>'], 'w')

    for k in range(10):
        out = []
        nw = random.sample(neg, quant)
        pw = random.sample(pos, quant)
        for i in range(quant):
            pieces = pw[i].split(' ')
            start = pieces[0]
            end = pieces[-1]
            out.append(f'{nw[i]} {start}')
            out.append(f'{end} {nw[quant-1-i]}')
        w.write('\n'.join(out))

# print the targets for a patent
if args['parse_tsv']:
    from parse_tsv import PatentTSV

    if args['--tsv']:
        print(PatentTSV(args['--tsv']).targets)
    else:
        tsvname = args['--xml'].replace("US0", "US")
        tsvname = tsvname[:tsvname.find("-")] + ".tsv"
        print(PatentTSV(tsvname).targets)

# output all targets from all tsvs
if args['parse_all_tsvs']:
    from os import path
    from glob import glob
    from parse_tsv import PatentTSV
    tsv_names = sorted(glob(path.join(args['<patents-dir>'], '*.tsv')))

    # handle directory errors
    if len(tsv_names) == 0:
        raise Exception(f"No patents in {arguments['--patents']}.")

    with open(args['<out-file>'], 'w') as w:
        for tsv in tsv_names:
            w.write('\n'.join(PatentTSV(tsv).targets) + '\n')

if args['parse_patent']:
    from parse_patent import XMLDoc
    doc = XMLDoc(args['<xml-file>'])
    if args['--title'] or args['--all']:
        print(doc.title)
    if args['--abstract'] or args['--all']:
        print(doc.abstract)
    if args['--references'] or args['--all']:
        print(doc.references)
    if args['--whole'] or args['--all']:
        print(doc.whole)

if args['parse_all_patents']:
    from os import path
    from glob import glob
    from unidecode import unidecode
    from parse_patent import XMLDoc
    patent_names = sorted(glob(path.join(args['<patents-dir>'], '*.XML')))

    # handle directory errors
    if len(patent_names) == 0:
        raise Exception(f"No patents in {arguments['--patents']}.")

    with open(args['<out-file>'], 'w') as w:
        for patent in patent_names:
            w.write(unidecode(XMLDoc(patent).whole) + '\n')

if args['split']:
    import os
    import random
    
    base = args['<in-file>'].split('.')[0]

    with open(args['<in-file>']) as f:
        lines = f.read().split('\n')
        random.shuffle(lines)

        # split data
        fifths = len(lines) // 5
        train = lines[:3*fifths]
        val = lines[3*fifths:4*fifths]
        test = lines[4*fifths:]

    # output data next to infiles
    # TODO consider writing in a different place
    with open(f'{base}.train', 'w') as w:
        w.write('\n'.join(train))
    with open(f'{base}.val', 'w') as w:
        w.write('\n'.join(val))
    with open(f'{base}.test', 'w') as w:
        w.write('\n'.join(test))

if args['permute']:
    from itertools import permutations
    with open(args['<in-file>']) as f, open(args['<out-file>'], 'w') as w:
        for line in f:
            if len(line.split(' ')) < 4:
                ps = [' '.join(list(x)) for x in list(permutations(line.strip().split(' ')))]
                w.write('\n'.join(ps) + '\n')
            else:
                w.write(line)

if args['stem']:
    from stem import PorterStemmer
    p = PorterStemmer()
    with open(args['<in-file>']) as f, open(args['<out-file>'], 'w') as w:
        words = f.read().split(' ')
        for word in words:
            word = word.strip()
            w.write(p.stem(word, 0, len(word) - 1) + ' ')

if args['case']:
    def capitalize(word):
        if len(word) == 0:
            return ''

        if word[0].isalpha():
            if len(word) > 1:
                return word[0].upper() + word[1:]
            else:
                return word[0].upper()
        else:
            return word

    def g(list_, cap):
        out = []
        for i in range(len(list_)):
            if cap[i]:
                out.append(capitalize(list_[i]))
            else:
                out.append(list_[i].lower())
        return ' '.join(out)

    with open(args['<in-file>']) as f, open(args['<out-file>'], 'w') as w:
        for line in f:
            line = line.strip()
            words = line.split(' ')
            n = len(words)
            if n < 4:
                for i in range(2**n):
                    cap = [False for _ in range(n)]
                    for j in range(n):
                        cap[j] = i % (2 ** (n - j)) >= (2 ** (n - j - 1)) 
                    w.write(g(words, cap) + '\n')
            w.write(line + '\n')

if args['cluster']:
    import sys
    import choose
    from os import path
    from glob import glob
    from validate import predict_siamese, get_siamese_model

    info_names = sorted(glob(path.join(args['<info-dir>'], '*.info')))

    # handle directory errors
    if len(info_names) == 0:
        raise Exception(f"No info patents in {args['<info-dir>']}.")

    model = get_siamese_model(args['<model.pkl>'], args['<device>'])

    with open(args['<out-file>'], 'w') as w:
        w.write("C, True Target, Cluster1, Cluster2\n")
        for info in info_names[:10]:
            targets = []
            seqs = []

            for line in open(info):
                if line.startswith('#'):
                    targets.append(line.strip()[2:])
                elif len(targets) > 0:
                    text, pred = line.split(' ~ ')
                    preds = [float(p) for p in pred.split(', ')]
                    if preds[1] > 0.5 and len(text) > 4:
                        seqs.append((text, preds[1]))

            # cluster seqs
            if len(seqs) > 0:
                clusters = choose.best_options(seqs, metric=predict_siamese(model), distance=0.05)
                if len(clusters) < 1:
                    clusters.append(([], 0))
                if len(clusters) < 2:
                    clusters.append(([], 0))
            
                w.write(f'x, {targets}, {clusters[0][0]}, {clusters[1][0]}\n')

if args['dedup']:
    pl = {}

    delim = args['<delimiter>']
    for line in open(args['<in-file>']):
        line = line.strip()
        names = line.split(delim)
        if names[0] not in pl:
            pl[names[0]] = []
        for n in names:
            if n not in pl[names[0]]:
                pl[names[0]].append(n)

    print(f'Unique items: {len(pl)}')
    with open(args['<out-file>'], 'w') as w:
        for key in pl:
            w.write(f"{delim.join(pl[key])}\n")

if args['lower']:
    delim = args['<delimiter>']
    with open(args['<in-file>']) as i, open(args['<out-file>'], 'w') as o:
        for line in i:
            line = line.strip()
            o.write(f'{line}{delim}{line.lower()}\n')
