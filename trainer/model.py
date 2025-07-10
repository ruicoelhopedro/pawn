import chess
import torch
import numpy as np
from torch import nn
from utils import map_features


NUM_FEATURES = 20480
NUM_ACCUMULATORS = 512
NUM_BUCKETS = 4


SCALE_FACTOR = 1024
NUM_MAX_FEATURES = 30


PIECE_VALUES = {
    chess.PAWN: (125, 200),
    chess.KNIGHT: (750, 850),
    chess.BISHOP: (800, 900),
    chess.ROOK: (1200, 1400),
    chess.QUEEN: (2500, 2600),
}


def sigmoid_loss(y, scores, results):
    K = 400
    mix = 0.3
    y_wdl = torch.sigmoid(SCALE_FACTOR / K * y)
    scores_wdl = (1 - mix) * torch.sigmoid(scores / K) + mix * results
    loss = (
        torch.abs(y_wdl - scores_wdl) +
        0.1 * torch.abs(torch.sigmoid((y - scores) / K) - 0.5)
    )
    return torch.mean(torch.pow(loss, 2.5))


class CReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):

    def __init__(self):
        super().__init__()
        self.crelu = CReLU()
        self.psqt = nn.EmbeddingBag(NUM_FEATURES, NUM_BUCKETS, mode='sum')
        self.accumulator_emb = nn.EmbeddingBag(NUM_FEATURES, NUM_ACCUMULATORS, mode='sum')
        self.accumulator_bias = nn.Parameter(torch.zeros(NUM_ACCUMULATORS))
        self.layer = nn.Linear(2 * NUM_ACCUMULATORS, NUM_BUCKETS)
        # Clear weights and biases for sparse input layer and PSQT
        self.psqt.weight.data.fill_(0.0)
        self.accumulator_emb.weight.data.fill_(0.0)
        self.accumulator_bias.data.fill_(0.0)
        # Initialise PSQT weights using material balance
        for square in range(64):
            for king in range(64):
                for piece, (_, eg) in PIECE_VALUES.items():
                    self.psqt.weight.data[map_features(piece, square, king, True), :] = eg / SCALE_FACTOR

    def forward(self, w_offset, w_cols, b_offset, b_cols, buckets):
        psqt = self.psqt(w_cols, w_offset) - self.psqt(b_cols, b_offset)
        stm_acc = self.crelu(self.accumulator_emb(w_cols, w_offset) + self.accumulator_bias)
        ntm_acc = self.crelu(self.accumulator_emb(b_cols, b_offset) + self.accumulator_bias)
        positional = self.layer(torch.cat([stm_acc, ntm_acc], dim=1))
        bucketed_output = psqt + positional
        return bucketed_output.gather(-1, buckets.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def __dump(tensor, dtype, scale, file, transpose=False):
        weights = tensor.detach().to('cpu').numpy()
        quant_weights = np.array(np.round(scale * weights), dtype=dtype)
        if transpose:
            quant_weights = quant_weights.T
        quant_weights.tofile(file)

    def export(self, filename):
        # Export each layer to the output NNUE file
        with open(filename, 'w') as output_file:
            self.__dump(self.accumulator_emb.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.psqt.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.accumulator_bias.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.layer.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.layer.bias.data, np.short, SCALE_FACTOR, output_file)
