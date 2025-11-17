import chess
import torch
import numpy as np
from torch import nn

NUM_SQUARES = 64
NUM_PIECE_TYPES = 6
NUM_COLORS = 2
NUM_INPUT_BUCKETS = 32
NUM_FEATURES = NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS * NUM_INPUT_BUCKETS
NUM_ACCUMULATORS = 512
NUM_OUTPUT_BUCKETS = 4


SCALE_FACTOR = 1024
NUM_MAX_FEATURES = 32


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
        self.psqt = nn.EmbeddingBag(NUM_FEATURES, NUM_OUTPUT_BUCKETS, mode='sum')
        self.accumulator = nn.EmbeddingBag(NUM_FEATURES, NUM_ACCUMULATORS, mode='sum')
        self.layer = nn.Linear(2 * NUM_ACCUMULATORS, NUM_OUTPUT_BUCKETS)
        # Clear weights and biases for sparse input layer and PSQT
        self.psqt.weight.data.fill_(0.0)
        self.accumulator.weight.data.fill_(0.0)
        # Initialise PSQT weights using material balance
        psqt_view = self.psqt.weight.data.view(
            NUM_INPUT_BUCKETS,
            NUM_COLORS,
            NUM_PIECE_TYPES,
            NUM_SQUARES,
            NUM_OUTPUT_BUCKETS,
        )
        for piece, (_, eg) in PIECE_VALUES.items():
            psqt_view[:, 0, piece - 1, :, :] = eg / (2 * SCALE_FACTOR)
            psqt_view[:, 1, piece - 1, :, :] = -eg / (2 * SCALE_FACTOR)

    def forward(self, w_offset, w_cols, b_offset, b_cols, buckets):
        psqt = self.psqt(w_cols, w_offset) - self.psqt(b_cols, b_offset)
        stm_acc = self.crelu(self.accumulator(w_cols, w_offset))
        ntm_acc = self.crelu(self.accumulator(b_cols, b_offset))
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
            self.__dump(self.accumulator.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.psqt.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.layer.weight.data, np.short, SCALE_FACTOR, output_file)
            self.__dump(self.layer.bias.data, np.short, SCALE_FACTOR, output_file)
