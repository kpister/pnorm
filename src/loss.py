import random
import torch 
from torch.nn import functional 
from torch.autograd import Variable 

class SimilarityLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        self.margin = margin

    # euclidean distance
    # harder negative mining thanks to jimmy yan
    def _forward(self, output1, output2, quant=100) -> torch.Tensor:
        # compute the positive loss of all the data
        pos_loss = torch.pow(torch.norm(output2.sub(output1), dim=1), 2)
        neg_loss = torch.empty_like(pos_loss)

        quant = min(quant, output1.size(0)-1)
        # find the hardest negative example
        for i, o_i in enumerate(output1):
            values, indices = torch.topk(torch.norm(output2.sub(o_i), dim=1), quant, largest=False)

            rn = int(random.random()*quant)
            while indices[rn] == i:
                rn = int(random.random()*quant)

            neg_loss[i] = self.margin - values[rn]

        neg_loss = torch.clamp(neg_loss, 0.0)

        return torch.mean(pos_loss) + torch.mean(neg_loss)

def MorphemeLoss():
    return torch.nn.NLLLoss()

# helper function for MaskedCrossEntropy
def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def MaskedCrossEntropy(logits, target, length):
    length = Variable(torch.LongTensor(length)).cuda()

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.

    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss
