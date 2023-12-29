import torch
import argparse
import numpy as np
from utils import build_features
from model import NNUE, SCALE_FACTOR


def main(net_file: str, fen: str):
    model = torch.load(net_file, map_location='cpu')
    w_offset, w_cols, b_offset, b_cols, buckets, stms = build_features(fen)
    print(model(w_offset, w_cols, b_offset, b_cols, buckets, stms) * SCALE_FACTOR / 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE file generator for pawn')
    parser.add_argument('model', help='PyTorch net model from the trainer')
    parser.add_argument('--fen', help='FEN string to evaluate (startpos by default)',
                        default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    args = parser.parse_args()
    main(args.model, args.fen)
