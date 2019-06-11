"""Generator
Usage:
    generator.py <in-file> <out-file> [--ngrams=<n>]

Options:
    --ngrams=<n>    set the size of the largest ngram to create [default: 3]
"""

# alg inspired by: http://www.albertauyeung.com/post/generating-ngrams-python/
def generate_ngrams(s, n=3):
    sequences = []
    for line in s.split('\n'):
        # Break sentence in the token, remove empty tokens
        tokens = [token for token in line.strip().split(" ") if token != ""]

        n_ = min(n, len(tokens)-1)
        while n_ > 0:
            # Use the zip function to help us generate n-grams
            # Concatentate the tokens into ngrams and return
            ngrams = zip(*[tokens[i:] for i in range(n_)])
            sequences += [" ".join(ngram) for ngram in ngrams]
            n_ -= 1
    return sequences


if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)

    with open(args['<in-file>']) as i, open(args['<out-file>'], 'w') as o:
        o.write('\n'.join(generate_ngrams(i.read(), n=int(args['--ngrams']))))
    
