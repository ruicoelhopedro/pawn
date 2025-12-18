"""NNUE model definition for chess evaluation."""
from typing import Optional
from io import TextIOWrapper
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


def dump(
    tensor: torch.Tensor, dtype: np.dtype, scale: int, file: TextIOWrapper
) -> None:
    """Dump a tensor to a file with quantization.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to dump.
    dtype : np.dtype
        Numpy data type to use for quantization.
    scale : int
        Scale factor for quantization.
    file : TextIOWrapper
        File to write the quantized tensor to.
    """
    weights = tensor.detach().to('cpu').numpy()
    quant_weights = np.array(np.round(scale * weights), dtype=dtype)
    quant_weights.tofile(file)


class NNUE(nn.Module):
    """Efficiently-updatable Neural Network (NNUE)."""

    def __init__(self) -> None:
        super().__init__()
        self.psqt = nn.EmbeddingBag(
            NUM_FEATURES, NUM_OUTPUT_BUCKETS, mode='sum'
        )
        self.accumulator = nn.EmbeddingBag(
            NUM_FEATURES, NUM_ACCUMULATORS, mode='sum'
        )
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

    @staticmethod
    def crelu(x: torch.Tensor) -> torch.Tensor:
        """Apply clipped ReLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated tensor.
        """
        return torch.clamp(x, 0.0, 1.0)

    def forward(
        self,
        w_offset: torch.Tensor,
        w_cols: torch.Tensor,
        b_offset: torch.Tensor,
        b_cols: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the NNUE.

        Parameters
        ----------
        w_offset : torch.Tensor
            Offsets for white pieces.
        w_cols : torch.Tensor
            Columns for white pieces.
        b_offset : torch.Tensor
            Offsets for black pieces.
        b_cols : torch.Tensor
            Columns for black pieces.
        buckets : torch.Tensor
            Output buckets.

        Returns
        -------
        torch.Tensor
            Computed NNUE scores.
        """
        psqt = self.psqt(w_cols, w_offset) - self.psqt(b_cols, b_offset)
        stm_acc = self.crelu(self.accumulator(w_cols, w_offset))
        ntm_acc = self.crelu(self.accumulator(b_cols, b_offset))
        positional = self.layer(torch.cat([stm_acc, ntm_acc], dim=1))
        bucketed_output = psqt + positional
        return bucketed_output.gather(-1, buckets.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def loss(
        y: torch.Tensor, scores: torch.Tensor, results: torch.Tensor
    ) -> torch.Tensor:
        """Loss function for NNUE training.

        Parameters
        ----------
        y : torch.Tensor
            Predicted scores.
        scores : torch.Tensor
            Target scores.
        results : torch.Tensor
            Game results (0: loss, 0.5: draw, 1: win).

        Returns
        -------
        torch.Tensor
            Computed loss.
        """
        K = 400
        mix = 0.3
        y_wdl = torch.sigmoid(SCALE_FACTOR / K * y)
        scores_wdl = (1 - mix) * torch.sigmoid(scores / K) + mix * results
        loss = (
            torch.abs(y_wdl - scores_wdl) +
            0.1 * torch.abs(torch.sigmoid((y - scores) / K) - 0.5)
        )
        return torch.mean(torch.pow(loss, 2.5))

    def export(self, filename: str) -> None:
        """Export the NNUE model to a file.

        Parameters
        ----------
        filename : str
            Path to the output file.
        """
        # Export each layer to the output NNUE file
        with open(filename, 'wb') as file:
            dump(self.accumulator.weight.data, np.short, SCALE_FACTOR, file)
            dump(self.psqt.weight.data, np.short, SCALE_FACTOR, file)
            dump(self.layer.weight.data, np.short, SCALE_FACTOR, file)
            dump(self.layer.bias.data, np.short, SCALE_FACTOR, file)

    def info(self) -> Optional[str]:
        """Return model information string.

        Returns
        -------
        Optional[str]
            Model information string or None if not applicable.
        """
        return None
