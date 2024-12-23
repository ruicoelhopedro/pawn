import os
import ctypes
import argparse
import subprocess
import numpy as np
from hashlib import sha256
from numpy.ctypeslib import ndpointer
import torch
import torch.sparse
from torch import nn
from tqdm import tqdm
from model import NNUE, NUM_MAX_FEATURES, sigmoid_loss


class PawnDataloader:

    def __init__(self, pawn_path: str):
        # Load and prepare the library
        self.lib = ctypes.CDLL(pawn_path)
        self.lib.init_pawn()

        # Set up all the required functions
        self.get_num_games = self.lib.get_num_games
        self.get_num_games.restype = ctypes.c_ulonglong
        self.get_num_games.argtypes = [ctypes.c_char_p]

        self.get_indices = self.lib.get_indices
        self.get_indices.argtypes = [ctypes.c_char_p,
                                     ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS")]

        self.get_num_positions = self.lib.get_num_positions
        self.get_num_positions.restype = ctypes.c_ulonglong
        self.get_num_positions.argtypes = [ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ctypes.c_ulonglong]

        self.get_nnue_data = self.lib.get_nnue_data
        self.get_nnue_data.restype = ctypes.c_ulonglong
        self.get_nnue_data.argtypes = [ctypes.c_char_p,
                                       ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                       ctypes.c_ulonglong,
                                       ctypes.c_ulonglong,
                                       ctypes.c_ulonglong,
                                       ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_ushort, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_short, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS"),
                                       ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS")]

    def load(self, dataset_path: str):
        # Load total number of games in the dataset
        fname = ctypes.create_string_buffer(bytes(dataset_path, 'utf-8'))
        n_games = self.get_num_games(ctypes.create_string_buffer(bytes(dataset_path, 'utf-8')))

        # Build the indices list
        indices = np.empty(n_games + 1, dtype=np.ulonglong)
        self.get_indices(fname, indices)

        # Get number of positions
        games = np.arange(n_games, dtype=np.ulonglong)
        n_max_pos = self.get_num_positions(indices, games, len(games))
        n_max_features = n_max_pos * NUM_MAX_FEATURES

        # Prepare all the storage
        w_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        b_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        w_cols = np.empty(n_max_features, dtype=np.uint16)
        b_cols = np.empty(n_max_features, dtype=np.uint16)
        scores = np.empty(n_max_pos, dtype=np.int16)
        results = np.empty(n_max_pos, dtype=np.byte)
        buckets = np.empty(n_max_pos, dtype=np.byte)
        stms = np.empty(n_max_pos, dtype=np.byte)
        # Read the games and get the actual number of positions
        n_pos = self.get_nnue_data(
            fname,
            indices,
            games,
            len(games),
            0,
            1,
            w_idx,
            b_idx,
            w_cols,
            b_cols,
            scores,
            results,
            buckets,
            stms,
        )
        # Build reduced arrays
        scores_array = torch.tensor(scores[0:n_pos], dtype=torch.float32)
        buckets_array = torch.tensor(buckets[0:n_pos], dtype=torch.long)
        results_array = (torch.tensor(results[0:n_pos], dtype=torch.float32) + 1) / 2
        stms_array = torch.tensor(stms[0:n_pos], dtype=torch.float32)
        # Build the embedding tensors
        w_cols =   torch.LongTensor(np.array(w_cols[:w_idx[n_pos]], dtype=np.int_))
        b_cols =   torch.LongTensor(np.array(b_cols[:b_idx[n_pos]], dtype=np.int_))
        w_offset = torch.LongTensor(np.array(w_idx[:n_pos], dtype=np.int_))
        b_offset = torch.LongTensor(np.array(b_idx[:n_pos], dtype=np.int_))
        return n_games, w_offset, w_cols, b_offset, b_cols, scores_array, results_array, buckets_array, stms_array


def generate_dataset(
    pawn_bin: str,
    nnue_file: str,
    num_threads: int,
    num_runs: int,
    depth: int,
    iteration: int,
    output_file: str,
) -> None:
    command = f"""setoption name NNUE_File value {nnue_file}
play_games depth {depth} runs_per_fen {num_runs} threads {num_threads} seed {iteration * num_threads} output_file {output_file}
quit"""
    subprocess.run([pawn_bin], input=command, text=True, check=True, stdout=subprocess.PIPE)


def train(
    dataloader: PawnDataloader,
    dataset_path: str,
    model: NNUE,
    loss_fn,
    optimiser,
    device,
    orig_model: NNUE,
) -> float:
    # Load data
    def send(x):
        return x.to(device)
    num_games, w_offset, w_cols, b_offset, b_cols, scores, results, buckets, stms = dataloader.load(dataset_path)
    w_offset, w_cols, b_offset, b_cols, scores, results, buckets, stms = map(send, (w_offset, w_cols, b_offset, b_cols, scores, results, buckets, stms))
    # Compute prediction error
    pred = model(w_offset, w_cols, b_offset, b_cols, stms)
    loss = loss_fn(pred.squeeze(), scores, results, buckets)
    # Backpropagation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    # Compute original loss
    with torch.no_grad():
        orig_pred = orig_model(w_offset, w_cols, b_offset, b_cols, stms)
        orig_loss = loss_fn(orig_pred.squeeze(), scores, results, buckets)
    return loss.item(), num_games, orig_loss.item()


def main(
    pawn_bin: str,
    pawn_lib: str,
    model_path: str,
    output_dir: str,
    iterations: int,
    num_threads: int,
    num_runs: int,
    save_every: int,
    depth: int,
) -> None:
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    print(f"Using {device} device")

    # Define model
    model: NNUE = torch.load(model_path, map_location=device)
    orig_model: NNUE = torch.load(model_path, map_location=device)

    # Loss and optimiser
    loss_func = sigmoid_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(output_dir, exist_ok=True)
    dataloader = PawnDataloader(pawn_lib)

    # Training loop
    with open('train.hist', 'w', encoding='utf8') as file:
        pbar = tqdm(range(iterations))
        for i in range(iterations):
            # Get the NNUE for this model
            nnue_file = os.path.join(output_dir, 'nnue.dat')
            model.export(nnue_file)

            hasher = sha256()
            with open(nnue_file, 'rb') as f:
                hasher.update(f.read())

            # Generate dataset for this iteration
            output_file = os.path.join(output_dir, f'output-{i}.dat')
            generate_dataset(pawn_bin, nnue_file, num_threads, num_runs, depth, i, output_file)

            # Make one training pass on the new dataset
            loss, num_games, orig_loss = train(dataloader, output_file, model, loss_func, optimiser, device, orig_model)

            # Dump results
            file.write(f'{i}\t{num_games}\t{loss:>10.6e}\t{orig_loss:>10.6e}\t{orig_loss - loss:>10.6e}\n')
            file.flush()
            pbar.set_postfix_str(f'Loss: {loss:>6.4e} ({num_games} games, improvement: {orig_loss - loss:>6.4e})')
            pbar.update()
            if i % save_every == 0:
                torch.save(model, os.path.join(output_dir, f"model-iter{i}"))
    torch.save(model, os.path.join(output_dir, "model"))
    pbar.close()

    # Save final model
    torch.save(model, os.path.join(output_dir, "model"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE trainer for pawn')
    parser.add_argument('pawn_bin', help='Binary of pawn', type=str)
    parser.add_argument('pawn_lib', help='Shared library of pawn', type=str)
    parser.add_argument('model_path', help='Initial model to start from', type=str)
    parser.add_argument('--output', type=str, default='reinforcement', help='Output directory')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads')
    parser.add_argument('--num_runs', type=int, default=32, help='Number of runs per thread')
    parser.add_argument('--depth', type=int, default=10, help='Depth to use during data generation')
    parser.add_argument('--save_every', type=int, default=8, help='Save the net every x iterations')
    args = parser.parse_args()
    main(args.pawn_bin, args.pawn_lib, args.model_path, args.output, args.iterations, args.num_threads, args.num_runs, args.save_every, args.depth)
