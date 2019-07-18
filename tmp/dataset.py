import os
import math
import torch #type: ignore
import random
from torch.utils.data import Dataset #type: ignore
from constants import SOS_token, EOS_token, VOCAB_SIZE

from typing import List, Tuple, Dict

tag_dict:Dict[str,int] = {}
tag_dict['end'] = VOCAB_SIZE

# convert any sequence to a tensor of the ascii values
def seq2tensor(seq: str) -> torch.Tensor:
    return torch.tensor([SOS_token] + [ord(c) for c in seq] + [EOS_token], dtype=torch.long)

# add tags to the global dictionary, return the tag index
def tag2idx(tag: str) -> int:
    if tag not in tag_dict:
        tag_dict[tag] = tag_dict['end']
        tag_dict['end'] += 1
    return tag_dict[tag]

# convert tag to tensor -- uses tag2idx
def tag2tensor(tag: str) -> torch.Tensor:
    pieces: List[str] = tag.split(';')
    return torch.tensor([tag2idx(p) for p in pieces])

# create a list of tuples -- tensor of the leader and tensor of each other member
def pairs(prot_list: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    leader, new_prot_list = seq2tensor(prot_list[0]), [seq2tensor(i) for i in prot_list[1:]]
    return [(leader, p) for p in new_prot_list]

class LoadData(Dataset):
    def __init__(self, filename:str, batch_size:int=1000) -> None:
        super(LoadData, self).__init__()
        assert os.path.isfile(filename)
        self.batch_size = batch_size
        self.data = self.load(filename)

    # to be implemented by child classes
    def load(self, fn:str) -> List:
        raise NotImplementedError

    # shuffle the dataset in place
    def shuffle(self) -> None:
        random.shuffle(self.data)

    # return the number of batches that this dataset contains
    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    # get a batch of data, size of batch is determined by batch_size
    def __getitem__(self, index:int) -> List[Tuple[torch.Tensor,...]]:
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.data))
        assert start < stop

        return self.data[start:stop]

class MorphemeData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000) -> None:
        super(MorphemeData, self).__init__(filename, batch_size)

    def load(self, fn:str) -> List[Tuple[torch.Tensor,...]]:
        # each line is target\tform\ttags
        data:List[Tuple[torch.Tensor,...]] = []
        for idx, line in enumerate(open(fn)):
            try:
                lemma, word, tags = line.split('\t')
                self.data.append((seq2tensor(lemma), 
                                  seq2tensor(word), 
                                  tag2tensor(tags)))
            except:
                print(f'Trouble parsing morph file on line {idx}')
        return data

class ProteinData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000):
        super(ProteinData, self).__init__(filename, batch_size)

    def load(self, fn:str) -> List[Tuple[torch.Tensor,...]]:
        # classes[index] :: list of names for the protein indexed by index
        # each class is a different protein
        # pos is a list of all positive pairs
        self.classes:Dict[int,List[str]] = {}
        data:List[Tuple] = []

        for idx, line in enumerate(open(fn)):
            prots = [i.strip() for i in line.split('~') if len(i.strip()) > 0]
            if len(prots) == 0 or len(prots) == 1: continue
            self.classes[idx] = prots
            data += pairs(prots)
        return data

    # for auc validation testing
    def get_neg(self, quant:int) -> Tuple[List, List]:
        p1 = []
        p2 = []
        for i in range(quant):
            leader, follow = random.sample(list(self.classes), 2)

            p1.append(seq2tensor(self.classes[leader][0]))
            p2.append(seq2tensor(random.choice(self.classes[follow][1:])))

        return p1, p2
