import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import NNUE, SCALE_FACTOR
from utils import map_features, rank, file


FILES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']
PIECES = ['', 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
COLORS = ['White', 'Black']


def square_str(square):
    file = int(square % 8)
    rank = int(square / 8)
    return f'{FILES[file]}{RANKS[rank]}'


def map_matrix(weights):
    matrix = np.zeros((8, 8))
    for i in range(8):
        row = (7-i)
        matrix[i, :] = weights[8*row : 8*(row+1)]
    return matrix


def plot_heatmap(data_white, data_black, king_sq, piece):
    # Map the data to a matrix
    w_matrix_mg = map_matrix(data_white[:, 0])
    w_matrix_eg = map_matrix(data_white[:, 1])
    b_matrix_mg = map_matrix(data_black[:, 0])
    b_matrix_eg = map_matrix(data_black[:, 1])
    # Create the plot
    fig, axes = plt.subplots(figsize=(8, 8), ncols=2, nrows=2)
    for row in axes:
        for ax in row:
            ax.set_xticks(np.arange(8))
            ax.set_yticks(np.arange(8))
            ax.set_xticklabels(FILES)
            ax.set_yticklabels(reversed(RANKS))
    axes[0, 0].imshow(w_matrix_mg, cmap='Blues', interpolation='nearest')
    axes[0, 1].imshow(w_matrix_eg, cmap='Blues', interpolation='nearest')
    axes[1, 0].imshow(b_matrix_mg, cmap='Blues', interpolation='nearest')
    axes[1, 1].imshow(b_matrix_eg, cmap='Blues', interpolation='nearest')
    for i in range(8):
        for j in range(8):
            axes[0, 0].text(j, i, int(w_matrix_mg[i, j]), ha="center", va="center", color="k")
            axes[0, 1].text(j, i, int(w_matrix_eg[i, j]), ha="center", va="center", color="k")
            axes[1, 0].text(j, i, int(b_matrix_mg[i, j]), ha="center", va="center", color="k")
            axes[1, 1].text(j, i, int(b_matrix_eg[i, j]), ha="center", va="center", color="k")
    for row in axes:
        for ax in row:
            rect = patches.Rectangle((file(king_sq) - 0.5, 7 - rank(king_sq) - 0.5), 1.0, 1.0, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    axes[0, 0].set_title('White perspective MG')
    axes[0, 1].set_title('White perspective EG')
    axes[1, 0].set_title('Black perspective MG')
    axes[1, 1].set_title('Black perspective EG')
    fig.suptitle(f'{PIECES[piece]} table for white king in {square_str(king_sq)}')
    fig.tight_layout()
    plt.show()



def main(net_file, piece, king_sq):
    model = torch.load(net_file, map_location='cpu')
    # Get the positions to fetch
    idx_white = [map_features(piece, square, king_sq, True) for square in range(64)]
    idx_black = [map_features(piece, square, king_sq, False) for square in range(64)]
    # Plot the heatmap
    weights_white = model.psqt.weight.data[idx_white,:] * SCALE_FACTOR
    weights_black = model.psqt.weight.data[idx_black,:] * SCALE_FACTOR
    plot_heatmap(weights_white, weights_black, king_sq, piece)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE data visualiser for pawn')
    parser.add_argument('model', help='PyTorch net model from the trainer')
    parser.add_argument('piece', help='Piece index to plot', type=int)
    parser.add_argument('square', help='Square index of the king', type=int)
    args = parser.parse_args()
    main(args.model, args.piece, args.square)
