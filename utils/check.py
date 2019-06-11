from itertools import permutations

def get_master():
    with open('text/pos.txt') as f:
        master = f.read().split('\n')
    res = {}
    for m in master:
        m = m.lower()
        if len(m) == 0:
            continue
        if m[0] not in res:
            res[m[0]] = []
        res[m[0]].append(m)

    return res

def check_word(word, master):
    matches = []
    if word == None or len(word) == 0:
        return matches

    # compare everything in lower case
    word = word.lower()

    if word[0] in master and word in master[word[0]]:
        matches.append(word)

    for perm in [' '.join(list(x)) for x in list(permutations(word.split(' ')))]:
        if len(perm) == 0:
            continue
        if perm[0] in master and perm in master[perm[0]] and perm not in matches:
            matches.append(perm)

    return matches

def check_batch(seqs, master):
    return [check_word(seq, master) for seq in seqs]
