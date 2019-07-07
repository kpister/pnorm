import os
import torch
import random
from torch.utils.data import Dataset

def pairs(prot_list):
    prot_list = [torch.tensor([ord(c) for c in i],dtype=torch.long) for i in prot_list]
    leader, prot_list = prot_list[0], prot_list[1:]
    return [(leader, p) for p in prot_list]

class ProteinData(Dataset):
    def __init__(self, filename):
        super(ProteinData, self).__init__()
        if isinstance(filename, str):
            filename = [filename]

        assert all([os.path.isfile(fn) for fn in filename])
        self.pos, self.classes, self.num_classes = self.load(filename)

    def load(self, filenames):
        # classes[index] :: list of names for the protein indexed by index
        # each class is a different protein
        # pos is a list of all positive pairs
        classes = {}
        pos = []

        # index represents the current class
        index = 0
        for fn in filenames:
            for line in open(fn):
                prots = [i.strip() for i in line.split('~') if len(i.strip()) > 0]
                if len(prots) == 0 or len(prots) == 1: continue
                classes[index] = prots
                pos += pairs(classes[index])
                index += 1
        return pos, classes, index

    def shuffle(self):
        random.shuffle(self.pos)

    def __len__(self):
        return len(self.pos)

    def get_neg(self, quant):
        p1 = []
        p2 = []
        for i in range(quant):
            pclass = pclass2 = random.randint(0, self.num_classes - 1)
            while pclass == pclass2:
                pclass2 = random.randint(0, self.num_classes - 1)

            p1.append(torch.tensor([ord(c) for c in (self.classes[pclass][0])],dtype=torch.long))
            p2.append(torch.tensor([ord(c) for c in (random.choice(self.classes[pclass2][1:]))],dtype=torch.long))
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

