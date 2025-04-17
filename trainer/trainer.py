import os
import ctypes
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from numpy.ctypeslib import ndpointer
import torch
import torch.sparse
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from model import NNUE, NUM_MAX_FEATURES, sigmoid_loss


class BatchedDataLoader:

    def __init__(self, dataset, batch_size, num_threads):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)

    def __iter__(self):
        return self.sampler.__iter__()

    def load(self, indices):
        # If no workers, just load the data sequentially
        if self.num_threads == 0:
            return self.dataset[indices]
        # Otherwise, load the data in parallel
        split_indices = np.array_split(indices, self.num_threads)
        with ThreadPool(self.num_threads) as pool:
            results = list(pool.map(self.dataset.__getitem__, split_indices))
        # Join the results
        # Most fields only require concatenation, but offsets need to be adjusted
        # based on the length of the previous results
        w_offset = results[0][0]
        b_offset = results[0][2]
        w_num = results[0][1].shape[0]
        b_num = results[0][3].shape[0]
        for i in range(1, self.num_threads):
            w_offset = torch.cat([w_offset, results[i][0] + w_num], dim=0)
            b_offset = torch.cat([b_offset, results[i][2] + b_num], dim=0)
            w_num += results[i][1].shape[0]
            b_num += results[i][3].shape[0]
        w_cols = torch.cat([r[1] for r in results], dim=0)
        b_cols = torch.cat([r[3] for r in results], dim=0)
        scores_array = torch.cat([r[4] for r in results], dim=0)
        results_array = torch.cat([r[5] for r in results], dim=0)
        buckets_array = torch.cat([r[6] for r in results], dim=0)
        return w_offset, w_cols, b_offset, b_cols, scores_array, results_array, buckets_array

    def set_train(self):
        self.dataset.set_train()

    def set_test(self):
        self.dataset.set_test()


class PawnDataset(Dataset):

    def __init__(self, pawn_path: str, dataset_path: str, prob_skip, test_size):
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
                                       ndpointer(ctypes.c_byte, flags="C_CONTIGUOUS")]

        # Load total number of games in the dataset
        print('Counting games...')
        self.fname = ctypes.create_string_buffer(bytes(dataset_path, 'utf-8'))
        self.n_games = self.get_num_games(self.fname)

        # Build the indices list
        print('Building indices...')
        self.indices = np.empty(self.n_games + 1, dtype=np.ulonglong)
        self.get_indices(self.fname, self.indices)

        # Get number of positions
        games = np.arange(self.n_games, dtype=np.ulonglong)
        self.n_pos = self.get_num_positions(self.indices, games, len(games))
        print(f'Found {self.n_games} games with a total of {self.n_pos} positions')

        # Train-test split
        self.train_indices, self.test_indices = train_test_split(np.arange(self.n_games), test_size=test_size)
        self.train_games = len(self.train_indices)
        self.test_games = len(self.test_indices)
        self.train = True

    def set_train(self):
        self.train = True

    def set_test(self):
        self.train = False

    def __len__(self):
        return self.train_games if self.train else self.test_games

    def __getitem__(self, index):
        mapper = self.train_indices if self.train else self.test_indices
        games = np.array(mapper[index], dtype=np.ulonglong)
        # Find maximum number of positions we can get from these games
        n_max_pos = self.get_num_positions(self.indices, games, len(games))
        n_max_features = n_max_pos * NUM_MAX_FEATURES
        # Prepare all the storage
        w_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        b_idx = np.empty(n_max_pos + 1, dtype=np.ulonglong)
        w_cols = np.empty(n_max_features, dtype=np.uint16)
        b_cols = np.empty(n_max_features, dtype=np.uint16)
        scores = np.empty(n_max_pos, dtype=np.int16)
        results = np.empty(n_max_pos, dtype=np.byte)
        buckets = np.empty(n_max_pos, dtype=np.byte)
        # Read the games and get the actual number of positions
        n_pos = self.get_nnue_data(self.fname, self.indices, games, len(games), hash(str(index)), self.prob_skip,
                                   w_idx, b_idx, w_cols, b_cols, scores, results, buckets)
        # Build reduced arrays
        scores_array = torch.tensor(scores[0:n_pos], dtype=torch.float32)
        buckets_array = torch.tensor(buckets[0:n_pos], dtype=torch.long)
        results_array = (torch.tensor(results[0:n_pos], dtype=torch.float32) + 1) / 2
        # Build the embedding tensors
        w_cols =   torch.LongTensor(np.array(w_cols[:w_idx[n_pos]], dtype=np.int_))
        b_cols =   torch.LongTensor(np.array(b_cols[:b_idx[n_pos]], dtype=np.int_))
        w_offset = torch.LongTensor(np.array(w_idx[:n_pos], dtype=np.int_))
        b_offset = torch.LongTensor(np.array(b_idx[:n_pos], dtype=np.int_))
        return w_offset, w_cols, b_offset, b_cols, scores_array, results_array, buckets_array



def train(dataloader, model, loss_fn, optimiser, device, epoch, output_file):
    send = lambda x: x.to(device)
    model.train()
    dataloader.set_train()
    size = len(dataloader.dataset)
    n_batches = size // dataloader.batch_size + 1
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', total=n_batches, unit='batch')
    for batch, indices in enumerate(dataloader):
        w_offset, w_cols, b_offset, b_cols, scores, results, buckets = map(send, dataloader.load(indices))

        # Compute prediction error
        pred = model(w_offset, w_cols, b_offset, b_cols, buckets)
        loss = loss_fn(pred, scores, results)

        # Backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update progress
        output_file.write(f'{epoch}\t{batch}\t{loss.item()}\n')
        output_file.flush()
        pbar.set_postfix_str(f'Loss: {loss.item():>6.4e}')
        pbar.update()
    pbar.close()


def test(dataloader, model, loss_fn, device, epoch, output_file):
    send = lambda x: x.to(device)
    test_loss = 0
    num_batches = 0
    dataloader.set_test()
    with torch.no_grad():
        for indices in dataloader:
            w_offset, w_cols, b_offset, b_cols, scores, results, buckets = map(send, dataloader.load(indices))
            pred = model(w_offset, w_cols, b_offset, b_cols, buckets)
            test_loss += loss_fn(pred, scores, results).item()
            num_batches += 1
        test_loss /= num_batches
    output_file.write(f'{epoch}\t{test_loss}\n')
    output_file.flush()
    print(f'Epoch {epoch + 1}: Test loss: {test_loss:>6.4e}')




def main(pawn_path: str, dataset_path: str, output_dir: str, epochs: int, batch_size: int, random_skip: int, test_size: float, num_threads: int, load_model=None):
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    print(f"Using {device} device")

    # Build dataset and dataloaders
    dataset = PawnDataset(pawn_path, dataset_path, random_skip, test_size)
    dataloader = BatchedDataLoader(dataset, batch_size, num_threads)

    # Define model
    if load_model is None:
        model = NNUE()
        model = model.to(device)
    else:
        model = torch.load(load_model, map_location=device)

    # Loss and optimiser
    loss_func = sigmoid_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training time
    os.makedirs(output_dir, exist_ok=True)
    scheduler = StepLR(optimiser, step_size=30, gamma=0.65)
    with open('train.hist', 'w') as train_file, open('test.hist', 'w') as test_file:
        for epoch in range(epochs):
            train(dataloader, model, loss_func, optimiser, device, epoch, train_file)
            test(dataloader, model, loss_func, device, epoch, test_file)
            scheduler.step()
            torch.save(model, os.path.join(output_dir, f"model-epoch{epoch}"))

    # Save final model
    torch.save(model, os.path.join(output_dir, "model"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='NNUE trainer for pawn')
    parser.add_argument('dataset', help='Dataset file to use for training', type=str)
    parser.add_argument('pawn', help='Shared library of pawn', type=str)
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--load', type=str, default=None, help='Start from a given model')
    parser.add_argument('--batch_size', type=int, default=16384, help='Number of games per batch')
    parser.add_argument('--random_skip', type=int, default=8, help='On average, skip every x positions')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test dataset size')
    parser.add_argument('--num_threads', type=int, default=1, help='Test dataset size')
    args = parser.parse_args()
    main(args.pawn, args.dataset, args.output, args.epochs, args.batch_size, args.random_skip, args.test_size, args.num_threads, args.load)
