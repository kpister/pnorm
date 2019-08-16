import os
import math
import torch #type: ignore
import random
from torch.utils.data import Dataset #type: ignore
from constants import *
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple, Dict

tag_dict:Dict[str,int] = {}
tag_dict['end'] = 2

def pad(tensor, size):
    return torch.cat((tensor, torch.tensor([EOS_token]*(size - tensor.size(0)), dtype=torch.long)))
# convert any sequence to a tensor of the ascii values
def seq2tensor(seq: str) -> torch.Tensor:
    return torch.tensor([SOS_token] + [ord(c) for c in seq] + [EOS_token], dtype=torch.long)
def target2tensor(seq: str) -> torch.Tensor:
    return torch.tensor([ord(c) for c in seq], dtype=torch.long)

# add tags to the global dictionary, return the tag index
def tag2idx(tag: str) -> int:
    if tag not in tag_dict:
        tag_dict[tag] = tag_dict['end']
        tag_dict['end'] += 1
    return tag_dict[tag]

# convert tag to tensor -- uses tag2idx
def input2tensor(word:str, tag: str) -> torch.Tensor:
    pieces: List[str] = tag.split(';')
    return torch.cat((
        torch.tensor([SOS_token], dtype=torch.long),
        torch.tensor([tag2idx(p) for p in pieces], dtype=torch.long),
        target2tensor(word), 
        torch.tensor([EOS_token], dtype=torch.long)
        ))

# create a list of tuples -- tensor of the leader and tensor of each other member
def pairs(prot_list: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    leader, new_prot_list = seq2tensor(prot_list[0]), [seq2tensor(i) for i in prot_list[1:]]
    return [(leader, p) for p in new_prot_list]

class LoadData(Dataset):
    def __init__(self, filename:str, batch_size:int=1000, empty=False) -> None:
        super(LoadData, self).__init__()
        self.batch_size = batch_size
        self.data: List[Tuple] = [] 

        if not empty:
            assert os.path.isfile(filename)
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
    def __getitem__(self, index:int) -> Tuple[torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor]:
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.data))
        assert start < stop

        left_max_len = max(self.data[start:stop], key=lambda v: len(v[0]))[0].size(0)
        right_max_len = max(self.data[start:stop], key=lambda v: len(v[1]))[1].size(0)

        left, right = [], []
        left_lens, right_lens = [], []
        for l, r in sorted(self.data[start:stop], key=lambda v: len(v[1]), reverse=True):
            left.append(pad(l, left_max_len))
            right.append(pad(r, right_max_len))
            left_lens.append(l.size(0))
            right_lens.append(r.size(0))

        return torch.stack(left), torch.tensor(left_lens), torch.stack(right), torch.tensor(right_lens)

class ParaData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000, empty=False) -> None:
        super(ParaData, self).__init__(filename, batch_size, empty)
        self.max_length = 500

    def load(self, fn:str) -> List[Tuple[torch.Tensor,torch.Tensor]]:
        data:List[Tuple[torch.Tensor,torch.Tensor]] = []
        for idx, line in enumerate(open(fn)):
            line = line.strip()
            try:
                s = [i.strip() for i in line.split('|||')]
                tag, p1, p2 = s[0], s[1], s[2]
                data.append((pad(target2tensor(p1), PARA_MAX_LENGTH),
                             seq2tensor(p2)))
            except Exception as e:
                print(f'Trouble parsing Paraphrase file on line {idx}: {e}')

            if idx > 100000:
                break
        return data
    
    def __len__(self):
        return min(self.max_length, super(ParaData, self).__len__())

class AcronymData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000, empty=False) -> None:
        super(AcronymData, self).__init__(filename, batch_size, empty)

    def load(self, fn:str) -> List[Tuple[torch.Tensor,torch.Tensor]]:
        # each line is acronym~expansion
        data:List[Tuple[torch.Tensor,torch.Tensor]] = []
        for idx, line in enumerate(open(fn)):
            line = line.strip()
            try:
                acro, expansion = line.split('~')
                data.append((pad(target2tensor(acro), ACRO_MAX_LENGTH), 
                             seq2tensor(expansion)))

            except Exception as e:
                print(f'Trouble parsing acronym file on line {idx}: {e}')
        return data

class MorphemeData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000, empty=False) -> None:
        super(MorphemeData, self).__init__(filename, batch_size, empty)
        self.num_tags = len(tag_dict) - 1

    def load(self, fn:str) -> List[Tuple[torch.Tensor,torch.Tensor]]:
        # each line is target\tform\ttags
        data:List[Tuple[torch.Tensor,torch.Tensor]] = []
        for idx, line in enumerate(open(fn)):
            line = line.strip()
            try:
                lemma, word, tags = line.split('\t')
                data.append((seq2tensor(lemma), 
                             input2tensor(word, tags)))

            except Exception as e:
                print(f'Trouble parsing morph file on line {idx}: {e}')
        return data

class ProteinData(LoadData):
    def __init__(self, filename:str, batch_size:int=1000, empty=False) -> None:
        super(ProteinData, self).__init__(filename, batch_size, empty)

    def load(self, fn:str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        # classes[index] :: list of names for the protein indexed by index
        # each class is a different protein
        # pos is a list of all positive pairs
        self.classes:Dict[int,List[str]] = {}
        data:List[Tuple[torch.Tensor, torch.Tensor]] = []

        for idx, line in enumerate(open(fn)):
            prots = [i.strip() for i in line.split('~') if len(i.strip()) > 0]
            if len(prots) == 0 or len(prots) == 1: continue
            self.classes[idx] = prots
            data += pairs(prots)
        return data

    # for auc validation testing
    def get_neg(self, quant:int) -> Tuple:
        p1 = []
        p2 = []
        for i in range(quant):
            leader, follow = random.sample(list(self.classes), 2)

            p1.append(seq2tensor(self.classes[leader][0]))
            p2.append(seq2tensor(random.choice(self.classes[follow][1:])))

        left_max_len = max(p1, key=lambda v: len(v)).size(0)
        right_max_len = max(p2, key=lambda v: len(v)).size(0)

        left, right = [], []
        left_lens, right_lens = [], []
        left_ord = []
        left_tmp = []

        # sort by right hand side
        i = 0
        for l, r in sorted(zip(p1,p2), key=lambda v: v[1].size(0), reverse=True):
            left_tmp.append((l, i))
            right.append(pad(r, right_max_len))
            right_lens.append(r.size(0))
            i += 1

        # sort by left hand side
        for l, i in sorted(left_tmp, key=lambda v: len(v[0]), reverse=True):
            left.append(pad(l, left_max_len))
            left_ord.append(i)
            left_lens.append(l.size(0))


        return torch.stack(left), torch.tensor(left_lens), left_ord, torch.stack(right), torch.tensor(right_lens), [i for i in right]



    # get a batch of data, size of batch is determined by batch_size
    def __getitem__(self, index:int) -> List[Tuple[torch.Tensor,torch.Tensor]]:
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.data))
        assert start < stop

        left_max_len = max(self.data[start:stop], key=lambda v: len(v[0]))[0].size(0)
        right_max_len = max(self.data[start:stop], key=lambda v: len(v[1]))[1].size(0)

        left, right = [], []
        left_lens, right_lens = [], []
        left_ord = []
        left_tmp = []

        # sort by right hand side
        i = 0
        for l, r in sorted(self.data[start:stop], key=lambda v: len(v[1]), reverse=True):
            left_tmp.append((l, i))
            right.append(pad(r, right_max_len))
            right_lens.append(r.size(0))
            i += 1

        # sort by left hand side
        for l, i in sorted(left_tmp, key=lambda v: len(v[0]), reverse=True):
            left.append(pad(l, left_max_len))
            left_ord.append(i)
            left_lens.append(l.size(0))


        return torch.stack(left), torch.tensor(left_lens), left_ord, torch.stack(right), torch.tensor(right_lens), [i for i in range(len(right))]
