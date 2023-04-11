import os
import ctypes
import argparse
import numpy as np
from numpy.ctypeslib import ndpointer
import torch
import torch.sparse
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from model import PositionalNet, SCALE_FACTOR, NUM_FEATURES, NUM_MAX_FEATURES


class BatchedDataLoader:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)

    def __iter__(self):
        return self.sampler.__iter__()

    def load(self, indices):
        return self.dataset[indices]



class PawnDataset(Dataset):

    def __init__(self, pawn_path: str, dataset_path: str, prob_skip):
        # Load and prepare the library
        self.lib = ctypes.CDLL(pawn_path)
        self.lib.init_pawn()

        self.prob_skip = prob_skip

        # Set up all the required functions
        self.get_num_games = self.lib.get_num_games
        self.get_num_games.restype = ctypes.c_ulonglong
        self.get_num_games.argtypes = [ctypes.c_char_p]

        self.get_indices = self.lib.get_indices
        self.get_indices.argtypes = [ctypes.c_char_p,
                                     ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS")]

        self.get_num_positions = self.lib.get_num_positions
        self.get_num_positions.argtypes = [ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ctypes.c_ulonglong]

        self.get_positional_data = self.lib.get_positional_data
        self.get_positional_data.restype = ctypes.c_ulonglong
        self.get_positional_data.argtypes = [ctypes.c_char_p,
                                             ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                             ctypes.c_ulonglong,
                                             ctypes.c_ulonglong,
                                             ctypes.c_ulonglong,
                                             ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS")]

        # Load total number of games in the dataset
        print('Counting games...')
        self.fname = ctypes.create_string_buffer(bytes(dataset_path, 'utf-8'))
        self.n_games = self.get_num_games(self.fname)

        # Build the indices list
        print('Building indices...')
        self.indices = np.empty(self.n_games + 1, dtype=np.ulonglong)
        self.get_indices(self.fname, self.indices)
        print(f'Found {self.n_games} games with a total of {self.indices[-1]} positions')

    def __len__(self):
        return self.n_games

    def __getitem__(self, index):
        games = np.array(index, dtype=np.ulonglong)
        # Find maximum number of positions we can get from these games
        n_max_pos = self.get_num_positions(self.indices, games, len(games))
        n_max_features = n_max_pos * NUM_MAX_FEATURES
        # Prepare all the storage
        w_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        b_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        w_cols = np.empty(n_max_features, dtype=np.uint16)
        b_cols = np.empty(n_max_features, dtype=np.uint16)
        evals = np.empty(n_max_pos, dtype=np.int16)
        scores = np.empty(n_max_pos, dtype=np.int16)
        results = np.empty(n_max_pos, dtype=np.byte)
        phases = np.empty(n_max_pos, dtype=np.byte)
        # Read the games and get the actual number of positions
        filtered = self.indices[games]
        n_pos = self.get_positional_data(self.fname, filtered, len(filtered), hash(str(index)), self.prob_skip,
                                         w_idx, b_idx, w_cols, b_cols, evals, scores, results, phases)
        # Build reduced arrays
        evals_array = torch.tensor(evals[0:n_pos], dtype=torch.float32)
        scores_array = torch.tensor(scores[0:n_pos], dtype=torch.float32)
        phases_array = torch.tensor(phases[0:n_pos], dtype=torch.float32) / 64.0
        results_array = (torch.tensor(results[0:n_pos], dtype=torch.float32) + 1) / 2
        # Build the sparse matrix
        w_idx = np.array(w_idx[:n_pos+1], dtype=np.int_)
        b_idx = np.array(b_idx[:n_pos+1], dtype=np.int_)
        w_cols = np.array(w_cols[:w_idx[n_pos]], dtype=np.int_)
        b_cols = np.array(b_cols[:w_idx[n_pos]], dtype=np.int_)
        w_matrix = torch.sparse_csr_tensor(w_idx, w_cols, np.ones(w_idx[n_pos]), size=(n_pos, NUM_FEATURES), dtype=torch.float32)
        b_matrix = torch.sparse_csr_tensor(b_idx, b_cols, np.ones(b_idx[n_pos]), size=(n_pos, NUM_FEATURES), dtype=torch.float32)
        return w_matrix, b_matrix, evals_array, scores_array, results_array, phases_array



def train(dataloader, model, loss_fn, optimiser, device):
    send = lambda x: x.to(device)
    curr = 0
    size = len(dataloader.dataset)
    model.train()
    for batch, indices in enumerate(dataloader):
        wf, bf, evals, scores, results, phases = map(send, dataloader.load(indices))

        # Compute prediction error
        pred = model(wf, bf)
        loss = loss_fn(pred.squeeze(), evals, scores, results, phases)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        curr += len(indices)
        print(f"Batch {batch:<5} loss: {loss.item():>4.3e}  [{curr:>7d}/{size:>7d}] ({100.0 * curr / size:>3.1f}%)")



def sigmoid_loss(output, evals, scores, results, phases):
    K = 256
    mix = 0.3
    y = output[:, 0] * phases + output[:, 1] * (1 - phases)
    y_wdl = torch.sigmoid((SCALE_FACTOR * y + evals) / K)
    scores_wdl = (1 - mix) * torch.sigmoid(scores / K) + mix * results
    return torch.mean((y_wdl - scores_wdl)**2)



def main(pawn_path: str, dataset_path: str, output_dir: str, epochs: int, batch_size: int, random_skip: int, load_model=None):
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    print(f"Using {device} device")

    # Build dataset and dataloaders
    dataset = PawnDataset(pawn_path, dataset_path, random_skip)
    dataloader = BatchedDataLoader(dataset, batch_size)

    # Define model
    if load_model is None:
        model = PositionalNet()
        model = model.to(device)
    else:
        model = torch.load(load_model, map_location=device)

    # Loss and optimiser
    loss_func = sigmoid_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training time
    os.makedirs(output_dir, exist_ok=True)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(dataloader, model, loss_func, optimiser, device)
        torch.save(model, os.path.join(output_dir, f"model-epoch{epoch}"))

    # Save final model
    torch.save(model, os.path.join(output_dir, "model"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE trainer for pawn')
    parser.add_argument('dataset', help='Dataset file to use for training', type=str)
    parser.add_argument('pawn', help='Shared library of pawn', type=str)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--load', type=str, default=None, help='Start from a given model')
    parser.add_argument('--batch_size', type=int, default=16384, help='Number of games per batch')
    parser.add_argument('--random_skip', type=int, default=15, help='On average, skip every x positions')
    args = parser.parse_args()
    main(args.pawn, args.dataset, args.output, args.epochs, args.batch_size, args.random_skip, args.load)
