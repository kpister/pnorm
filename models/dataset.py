import os
import torch
import random
from torch.utils.data import Dataset

vocab_size = 128
def lineToTensor(line):
    tensor = torch.zeros(len(line), vocab_size, dtype=torch.float)
    for li, letter in enumerate(line):
        tensor[li][ord(letter)] = 1
    return tensor

def pairs(prot_list):
    pairs = []
    pairs_char = []
    prot_list_char = prot_list
    prot_list_tensor = [lineToTensor(i) for i in prot_list]
    for i in range(1, len(prot_list)):
        #for k in range(i+1, len(prot_list)):
            #pairs.append((prot_list_tensor[i], prot_list_tensor[k]))
            #pairs_char.append((prot_list_char[i], prot_list_char[k]))
        # for few shot
        pairs.append((prot_list_tensor[0], prot_list_tensor[i]))
        pairs_char.append((prot_list_char[0], prot_list_char[i]))
    return pairs, pairs_char

#TODO build negative positive set. common words are close and far from everything else
# Add a single line of many non-protein ngrams
class ProteinData(Dataset):
    def __init__(self, filename, quant):
        super(ProteinData, self).__init__()
        if not os.path.isfile(filename):
            raise Exception(f'File {filename} does not exist')

        self.pos, self.pos_char, self.classes, self.num_classes = self.load(filename, quant)

    def load(self, filename, quant):
        # classes[index] :: list of names for the protein indexed by index
        # each class is a different protein
        # pos is a list of all positive pairs
        classes = {}
        pos = []
        pos_char = []

        # index represents the current class
        index = 0
        for line in open(filename):
            if index > quant and quant != -1:
                break
            q = [i.strip() for i in line.split('~') if len(i.strip()) > 0]
            if len(q) == 0 or len(q) == 1: continue
            classes[index] = q
            _pos, _pos_char = pairs(classes[index])
            pos += _pos
            pos_char += _pos_char
            index += 1
        return pos, pos_char, classes, index

    def shuffle(self):
        random.shuffle(self.pos)

    def __len__(self):
        return len(self.pos)

    def get_pos_char(self, index):
        # for hard negative mining
        if isinstance(index, int):
            index = slice(index,index+1,None)
        elif index.start: # else it is a slice
            index = slice(index.start, min(index.stop, len(self)), None)
        else:
            index = slice(0, min(index.stop, len(self)), None)

        p1 = []
        p2 = []
        for i, j in self.pos_char[index.start:index.stop]:
            p1.append(i)
            p2.append(j)
        return p1, p2


    def get_neg(self, quant):
        p1 = []
        p2 = []
        for i in range(quant):
            pclass = pclass2 = random.randint(0, self.num_classes - 1)
            while pclass == pclass2:
                pclass2 = random.randint(0, self.num_classes - 1)

            p1.append(lineToTensor(self.classes[pclass][0]))
            p2.append(lineToTensor(random.choice(self.classes[pclass2][1:])))
        return p1, p2

    def get_neg_char(self, quant):
        p1 = []
        p2 = []
        for i in range(quant):
            pclass = pclass2 = random.randint(0, self.num_classes - 1)
            while pclass == pclass2:
                pclass2 = random.randint(0, self.num_classes - 1)

            p1.append(random.choice(self.classes[pclass]))
            p2.append(random.choice(self.classes[pclass2]))
        return p1, p2


    # Return positive on even indices, negative on odd
    # Positive is deterministic, negative is random
    def __getitem__(self, index):
        # for hard negative mining
        if isinstance(index, int):
            if index >= len(self):
                raise Exception("Index out of bounds")
            index = slice(index,index+1,None)
        elif index.start: # else it is a slice
            index = slice(index.start, min(index.stop, len(self)), None)
            if index.stop <= index.start:
                return [], []
        else:
            index = slice(0, min(index.stop, len(self)), None)

        p1 = []
        p2 = []
        for i, j in self.pos[index.start:index.stop]:
            p1.append(i)
            p2.append(j)

        #p1, p2 = *self.pos[index.start:index.stop]
        return p1, p2

