import torch
from torch import nn


NUM_FEATURES = 20480
NUM_ACCUMULATORS = 32


SCALE_FACTOR = 1024
NUM_MAX_FEATURES = 30


class CReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class PositionalNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.psqt = nn.Linear(NUM_FEATURES, 2, bias=False)
        self.accumulator = nn.Linear(NUM_FEATURES, NUM_ACCUMULATORS)
        self.stack = nn.Sequential(CReLU(),
                                   nn.Linear(NUM_ACCUMULATORS, 2, bias=False))
        # Clear weights and biases for sparse input layer and PSQT
        self.psqt.weight.data.fill_(0.0)
        self.accumulator.weight.data.fill_(0.0)
        self.accumulator.bias.data.fill_(0.0)

    def forward(self, white, black):
        psqt = self.psqt(white) - self.psqt(black)
        positional = self.stack(self.accumulator(white)) - self.stack(self.accumulator(black))
        return psqt + positional
