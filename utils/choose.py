"""Choose Proteins
Usage: 
    choose.py <in-file> <distance>
"""

from sklearn.cluster import dbscan
from leven import levenshtein
import numpy as np
import torch

# process a .info file to extract the seqs for 'best_option'
def process_file(filename):
    # open file and read contents
    seqs = []

    # parse contents into sequences
    for line in open(filename):
        if '~' not in line:
            continue
        seq, pred = line.split(" ~ ")
        preds = [float(f) for f in pred.split(", ")]
        prot = preds[1]
        seqs.append((seq, prot))
    return seqs

# Seqs should be a list of tuples, (sequence, probability)
def best_options(seqs, metric=levenshtein, distance=10):
    data = [seq[0] for seq in seqs]
    X = np.arange(len(data)).reshape(-1, 1)

    def lev_metric(a, b):
        i, j = int(a[0]), int(b[0])
        return metric(data[i].lower(), data[j].lower())

    # TODO think about eps and min_samples
    cluster = dbscan(X, metric=lev_metric, eps=distance, min_samples=1, algorithm='brute')
    count = max(cluster[1])

    new_clusters = []
    for i in range(count + 1):
        cluster_i = [k for k,c in zip(cluster[0], cluster[1]) if c == i]
        if len(cluster_i) == 0:
            continue

        cluster_seqs = [seqs[el] for el in cluster_i]
        sorted_cluster_seqs = sorted(cluster_seqs, key=lambda kv:kv[1], reverse=True)
        avg_score = sum([seqs[el][1] for el in cluster_i]) / len(cluster_i)
        new_clusters.append((sorted_cluster_seqs, avg_score))

    return sorted(new_clusters, key=lambda kv:kv[1], reverse=True)

# given a list of filenames compute the siamese embedded space of all the sequences
def build_truth(filename, model):
    truth = None
    truth_seqs = []
    for line in open(filename):
        if '~' not in line:
            continue
        truth_seqs.append(line.split('~')[0])

    batch_size = 10000
    for i in range(0,len(truth_seqs) + batch_size, batch_size):
        end = min(i+batch_size, len(truth_seqs))
        if end < i:
            break

        h = model.init_hidden(end-i)
        if truth is None:
            truth = model.forward_one(truth_seqs[i:end], h)
        else:
            truth = torch.cat((truth, model.forward_one(truth_seqs[i:end], h)), 0)
        print(f"Batch {i} completed\r", end="")

    return (truth, truth_seqs)

def mag(t):
    s = sum([i * i for i in t])

def closest(seqs, truth, model):
    best_is = []
    h = model.init_hidden(1)
    for seq in seqs:
        # compute siamese of seq
        embed = model.forward_one([seq], h)

        # compute matmul(seq, truth), which is just a dot product of every vector in truth with seq
        # truth size :: 500,000 x 100; seq size :: 100x1; dists size :: 500,000 x 1
        dists = mag(truth - embed)
        #dists = torch.mm(truth, embed)

        # find the closest one
        best = -1
        best_i = -1
        for i, d in enumerate(truth):
            dist = mag(d - embed)
            if dist > best:
                best_i = i
                best = dist
        best_is.append(best_i)

    # return that
    return best_is


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    opts = best_options(process_file(args['<in-file>']), metric=levenshtein, distance=int(args['<distance>']))
    for i in opts:
        print(i)

