"""Preprocess
Usage: 
    preprocess.py <in-file> <out-file> <ngram-size>

"""

import random
from unidecode import unidecode
from generator import generate_ngrams
from stem import PorterStemmer

# convert word to ascii, remove none-internal non-alphanumeric characters
def preprocess_word(word):
    if word is None:
        return ''

    word = unidecode(word)
    end = -1
    start = 0
    for i, c in enumerate(reversed(word)):
        if c.isalnum() or c in [')',']']:
            end = len(word)-i
            break

    for i, c in enumerate(word):
        if c.isalnum() or c in ['(','[']:
            start = i
            break

    if end == -1:
        return ''

    return word[start:end]

# given a body of text, return:
# a list of <= 5-gram sequences s.t.
# remove all non-internal punctuation
# remove sequences of common words
# TODO consider cutting ngrams on each common word removed
def remove_common(text, common, ngram_size):
    p = PorterStemmer()
    extra_common = ['wherein', 'formula'] #TODO Always remove certain words
    # if three common words are next to each other, remove the inner one
    total = []

    for line in text.split('\n'):
        words = [preprocess_word(w) for w in line.split(' ')]
        comm  = [(word.lower() in common or p.stem(word.lower(), 0, len(word.lower()) - 1) in common or word == '') for word in words]

        new_text = []
        for i in range(len(words)):
            if comm[i]:
                if not ((i == 0 or comm[i-1]) and (i == len(comm) - 1 or comm[i+1])) and words[i].lower() not in extra_common:
                    #new_text.append(p.stem(words[i], 0, len(words[i]) - 1))
                    new_text.append(words[i])
                else:
                    # reset new text and generate ngrams
                    total += generate_ngrams(' '.join(new_text), ngram_size)
                    new_text = []
            else:
                new_text.append(words[i])

        total += generate_ngrams(' '.join(new_text), ngram_size)

    total = [i for i in total if i not in common]

    return total

# given a body of text return:
# a list of <= 5-gram sequences s.t.
# all non-internal punctuation removed
def preprocess(text, ngram_size):
    total = []

    for line in text.split("\n"):
        ## Remove non-internal punctuation
        words = [preprocess_word(w) for w in line.split(' ')]

        ## Convert to ngram token sequences
        # ngrams :: [sequences]
        total += generate_ngrams(' '.join(words), ngram_size)

    return total

if __name__ == '__main__':
    from docopt import docopt
    args = docopt(__doc__)
    with open(args['<in-file>']) as i, open(args['<out-file>'], 'w') as o:
        text = i.read()
        batch_size = 10000
        for i in range(0, len(text)+batch_size, batch_size):
            end = min(i+batch_size, len(text))
            ngrams  = preprocess(text[i:i+end], int(args['<ngram-size>']))
            o.write('\n'.join(ngrams))
