"""
Contrastive losses for the image encoder.

The original project uses InfoNCE with in-batch negatives, having tried triplet
loss first and found it worse. That result is expected: triplet loss gives one
negative per step, in-batch contrastive gives batch_size - 1, and contrastive
learning is bottlenecked almost entirely by negative count.

Two additions here:

  SigLIP loss   — sigmoid instead of softmax. Softmax requires normalising over
                  the whole batch, which forces an all-gather across GPUs and
                  caps how large a batch you can afford. Sigmoid scores each pair
                  independently, so batches can grow far larger, which for
                  contrastive learning is the thing that actually helps.

  Matryoshka    — compute the same loss at several truncations of the embedding
                  (64, 128, 256, 512) and sum them. The model is forced to put
                  the most important information in the earliest dimensions, so
                  one trained model serves cheap 64-d recall AND precise 512-d
                  reranking. No second model, no distillation step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Standard in-batch contrastive loss — what the original project used."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        logits = a @ p.T / self.t                       # [B, B]
        targets = torch.arange(len(a), device=a.device)  # diagonal is correct
        # Symmetric: anchor->positive and positive->anchor.
        return 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.T, targets))


class SigLIPLoss(nn.Module):
    """Pairwise sigmoid loss. Scales to much larger batches than InfoNCE."""

    def __init__(self, init_temperature: float = 0.07, init_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())
        self.logit_bias = nn.Parameter(torch.tensor(init_bias))

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        logits = a @ p.T * self.logit_scale.exp() + self.logit_bias
        n = len(a)
        # +1 on the diagonal (matching pairs), -1 everywhere else.
        labels = 2 * torch.eye(n, device=a.device) - 1
        return -F.logsigmoid(labels * logits).sum() / n


class MatryoshkaWrapper(nn.Module):
    """
    Apply a base loss at several embedding truncations at once.

    Weighting the shorter dimensions slightly higher matters: they are the ones
    doing the heavy lifting in the recall stage, where you are scanning the whole
    catalogue, and they are the harder problem because they have less room.
    """

    def __init__(self, base_loss: nn.Module, dims=(64, 128, 256, 512)):
        super().__init__()
        self.base = base_loss
        self.dims = sorted(dims)
        total = sum(1.0 / d ** 0.5 for d in self.dims)
        self.weights = [(1.0 / d ** 0.5) / total for d in self.dims]

    def forward(self, anchor, positive):
        loss = 0.0
        for d, w in zip(self.dims, self.weights):
            loss = loss + w * self.base(anchor[:, :d], positive[:, :d])
        return loss


class HardNegativeInfoNCE(nn.Module):
    """
    InfoNCE plus explicitly mined hard negatives.

    After a few epochs, in-batch negatives become trivially easy — a random other
    product in the batch is usually a completely different garment. Progress
    stalls. Mined hard negatives are the items the CURRENT model ranks highly but
    which are wrong, so they sit exactly on the decision boundary.

    Mine them from the search index itself: for each anchor, retrieve the top 50,
    drop the true positive, and keep what remains. Re-mine every couple of epochs,
    because what counts as hard changes as the model improves.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, anchor, positive, hard_negatives):
        # hard_negatives: [B, N, D]
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        h = F.normalize(hard_negatives, dim=-1)

        in_batch = a @ p.T / self.t                                   # [B, B]
        hard = torch.einsum("bd,bnd->bn", a, h) / self.t              # [B, N]
        logits = torch.cat([in_batch, hard], dim=1)                   # [B, B+N]
        targets = torch.arange(len(a), device=a.device)
        return F.cross_entropy(logits, targets)
