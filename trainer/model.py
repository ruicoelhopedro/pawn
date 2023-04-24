import chess
import torch
from torch import nn
from utils import map_features


NUM_FEATURES = 20480
NUM_ACCUMULATORS = 128


SCALE_FACTOR = 1024
NUM_MAX_FEATURES = 30


PIECE_VALUES = {
    chess.PAWN: (125, 200),
    chess.KNIGHT: (750, 850),
    chess.BISHOP: (800, 900),
    chess.ROOK: (1200, 1400),
    chess.QUEEN: (2500, 2600),
}


class CReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):

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
        # Initialise PSQT weights using material balance
        for square in range(64):
            for king in range(64):
                for piece, (mg, eg) in PIECE_VALUES.items():
                    self.psqt.weight.data[0, map_features(piece, square, king, True)] = mg / SCALE_FACTOR
                    self.psqt.weight.data[1, map_features(piece, square, king, True)] = eg / SCALE_FACTOR

    def forward(self, white, black):
        psqt = self.psqt(white) - self.psqt(black)
        positional = self.stack(self.accumulator(white)) - self.stack(self.accumulator(black))
        return psqt + positional
