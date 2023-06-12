import os
import ctypes
import argparse
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
from model import NNUE, SCALE_FACTOR, NUM_FEATURES, NUM_MAX_FEATURES


class BatchedDataLoader:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)

    def __iter__(self):
        return self.sampler.__iter__()

    def load(self, indices):
        return self.dataset[indices]

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
        self.get_num_positions.argtypes = [ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                                           ctypes.c_ulonglong]

        self.get_nnue_data = self.lib.get_nnue_data
        self.get_nnue_data.restype = ctypes.c_ulonglong
        self.get_nnue_data.argtypes = [ctypes.c_char_p,
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
        phases = np.empty(n_max_pos, dtype=np.byte)
        # Read the games and get the actual number of positions
        filtered = self.indices[games]
        n_pos = self.get_nnue_data(self.fname, filtered, len(filtered), hash(str(index)), self.prob_skip,
                                   w_idx, b_idx, w_cols, b_cols, scores, results, phases)
        # Build reduced arrays
        scores_array = torch.tensor(scores[0:n_pos], dtype=torch.float32)
        phases_array = torch.tensor(phases[0:n_pos], dtype=torch.float32) / 64.0
        results_array = (torch.tensor(results[0:n_pos], dtype=torch.float32) + 1) / 2
        # Build the embedding tensors
        w_cols =   torch.LongTensor(np.array(w_cols[:w_idx[n_pos]], dtype=np.int_))
        b_cols =   torch.LongTensor(np.array(b_cols[:b_idx[n_pos]], dtype=np.int_))
        w_offset = torch.LongTensor(np.array(w_idx[:n_pos], dtype=np.int_))
        b_offset = torch.LongTensor(np.array(b_idx[:n_pos], dtype=np.int_))
        return w_offset, w_cols, b_offset, b_cols, scores_array, results_array, phases_array



def train(dataloader, model, loss_fn, optimiser, device, epoch, output_file):
    send = lambda x: x.to(device)
    model.train()
    dataloader.set_train()
    size = len(dataloader.dataset)
    n_batches = size // dataloader.batch_size + 1
    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', total=n_batches, unit='batch')
    for batch, indices in enumerate(dataloader):
        w_offset, w_cols, b_offset, b_cols, scores, results, phases = map(send, dataloader.load(indices))

        # Compute prediction error
        pred = model(w_offset, w_cols, b_offset, b_cols)
        loss = loss_fn(pred.squeeze(), scores, results, phases)

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
            w_offset, w_cols, b_offset, b_cols, scores, results, phases = map(send, dataloader.load(indices))
            pred = model(w_offset, w_cols, b_offset, b_cols)
            test_loss += loss_fn(pred.squeeze(), scores, results, phases).item()
            num_batches += 1
        test_loss /= num_batches
    output_file.write(f'{epoch}\t{test_loss}\n')
    output_file.flush()
    print(f'Epoch {epoch + 1}: Test loss: {test_loss:>6.4e}')



def sigmoid_loss(output, scores, results, phases):
    K = 400
    mix = 0.3
    y = output[:, 0] * phases + output[:, 1] * (1 - phases)
    y_wdl = torch.sigmoid(SCALE_FACTOR / K * y)
    scores_wdl = (1 - mix) * torch.sigmoid(scores / K) + mix * results
    return torch.mean((y_wdl - scores_wdl)**2)



def main(pawn_path: str, dataset_path: str, output_dir: str, epochs: int, batch_size: int, random_skip: int, test_size: float, load_model=None):
    # Get cpu or gpu device for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    print(f"Using {device} device")

    # Build dataset and dataloaders
    dataset = PawnDataset(pawn_path, dataset_path, random_skip, test_size)
    dataloader = BatchedDataLoader(dataset, batch_size)

    # Define model
    if load_model is None:
        model = NNUE()
        model = model.to(device)
    else:
        model = torch.load(load_model, map_location=device)

    # Loss and optimiser
    loss_func = sigmoid_loss
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)

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
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--load', type=str, default=None, help='Start from a given model')
    parser.add_argument('--batch_size', type=int, default=16384, help='Number of games per batch')
    parser.add_argument('--random_skip', type=int, default=15, help='On average, skip every x positions')
    parser.add_argument('--test_size', type=float, default=0.1, help='Test dataset size')
    args = parser.parse_args()
    main(args.pawn, args.dataset, args.output, args.epochs, args.batch_size, args.random_skip, args.test_size, args.load)
