import os
import torch
import shutil
import argparse
import numpy as np
from model import NNUE, SCALE_FACTOR
from hashlib import sha256


def main(net_file):
    model = torch.load(net_file, map_location='cpu')

    # Export NNUE to temporary file
    tmp_psqt_name = 'psqt.nn'
    model.export(tmp_psqt_name)

    # Update the hash of the file
    hasher = sha256()
    with open(tmp_psqt_name, 'rb') as file:
        hasher.update(file.read())
    psqt_name = f'nnue-{hasher.hexdigest()[0:12]}.dat'
    shutil.move(tmp_psqt_name, psqt_name)
    print(f'Exported net {psqt_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE file generator for pawn')
    parser.add_argument('model', help='PyTorch net model from the trainer')
    args = parser.parse_args()
    main(args.model)
