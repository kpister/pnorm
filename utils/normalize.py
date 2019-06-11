import torch
import torch.nn.functional as F

def write_embeddings(prot_list, model, out_file):
    with open(out_file, 'w') as w:
        for i in prot_list:
            e = model.batchless_one(i)
            w.write(f"{','.join([str(i.item()) for i in e[0]])}\n")

def get_prot_list(filename):
    res = []
    for line in open(filename):
        prot = line.split('~')[0]
        res.append(prot)
    return res

def get_siamese_model(filename, model_path, device='cuda:0'):
    import sys
    sys.path.append(model_path)
    from siamese_model import Siamese

    vocab_size = 128
    hidden_dim = 120
    device = torch.device(device)
    model = Siamese(vocab_size, hidden_dim, device)
    model.to(device)
    model.eval()
    return model

class Normalize:
    def __init__(self, prot_list, model):
        self.prot_list = sorted(prot_list, key=len, reverse=True)
        self.model = model

        #self.embedded_prots = self.embed_prots()

    def embed_prots(self):
        batch_size = 250
        embedding = []
        for i in range(0, len(self.prot_list)+batch_size, batch_size):
            end = min(i+batch_size, len(self.prot_list))
            if end <= i: 
                break

            res = self.model.forward_one(self.prot_list[i:end], 
                                         self.model.init_hidden(end-i))
            embedding += [i.to('cpu') for i in res]
        return embedding

    def __call__(self, *args, **kwargs):
        return self.normalize(*args)

    def normalize(self,seq):
        pairs = []
        embedding = self.model.batchless_one(seq)
        for protein in self.prot_list:
            prot_embed = self.model.batchless_one(protein)
            distance = F.pairwise_distance(embedding, prot_embed)
            pairs.append((protein, distance))
        closest = sorted(pairs, key=lambda x:x[1])[0]
        return closest
