"""Trainer for NNUE models using pawn datasets."""
from typing import List, Optional, TextIO, Literal
import os
import ctypes
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from numpy.ctypeslib import ndpointer
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from sklearn.model_selection import train_test_split
from simple_parsing import ArgumentParser
from model import NNUE, NUM_MAX_FEATURES
from pmap import ParallelMap


@dataclass
class Options:
    """Training options."""
    # Output directory for model checkpoints
    output_dir: str = 'models'

    # Number of epochs to train
    epochs: int = 300

    # Batch size for training
    batch_size: int = 16384

    # On average, accept 1 in every `random_skip` positions
    random_skip: int = 8

    # Fraction of data to use for testing
    test_size: float = 0.1

    # Number of threads for data loading
    num_threads: int = 1

    # Path to a model to load (if any)
    load: Optional[str] = None

    # Learning rate for the optimiser
    lr: float = 1e-3

    # Learning rate scheduler
    lr_scheduler: Literal['step', 'cosine', 'none'] = 'step'

    # Step LR scheduler parameters: gamma
    lr_step_gamma: float = 0.65

    # Step LR scheduler parameters: step size (in epochs)
    lr_step_size: int = 30

    # Cosine LR scheduler parameters: decay ratio
    lr_cosine_final: float = 1e-2

    # Interval (in epochs) to save model checkpoints
    save_interval: int = 10

    # Device to use for training
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class BatchedData:
    """Container for a batch of data."""

    w_offset: torch.LongTensor
    w_cols: torch.LongTensor
    b_offset: torch.LongTensor
    b_cols: torch.LongTensor
    scores: torch.Tensor
    results: torch.Tensor
    buckets: torch.Tensor

    def model_inputs(self) -> List[torch.Tensor]:
        """Return the model input tensors as a list."""
        return [
            self.w_offset,
            self.w_cols,
            self.b_offset,
            self.b_cols,
            self.buckets,
        ]

    def loss_targets(self) -> List[torch.Tensor]:
        """Return the loss target tensors as a list."""
        return [
            self.scores,
            self.results,
        ]

    def to(self, device: torch.device) -> 'BatchedData':
        """Move data to the specified device."""
        return BatchedData(
            self.w_offset.to(device),
            self.w_cols.to(device),
            self.b_offset.to(device),
            self.b_cols.to(device),
            self.scores.to(device),
            self.results.to(device),
            self.buckets.to(device),
        )


class PawnDataset(Dataset):
    """Dataset for loading pawn data using the provided shared library."""

    def __init__(
        self,
        pawn_path: str,
        dataset_path: str,
        prob_skip: int,
        test_size: float,
    ) -> None:
        # Load and prepare the library
        self.lib = ctypes.CDLL(pawn_path)
        self.lib.init_pawn()

        self.prob_skip = prob_skip

        # Set up all the required functions
        self.get_num_games = self.lib.get_num_games
        self.get_num_games.restype = ctypes.c_ulonglong
        self.get_num_games.argtypes = [ctypes.c_char_p]

        self.get_indices = self.lib.get_indices
        self.get_indices.argtypes = [
            ctypes.c_char_p,
            ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
        ]

        self.get_num_positions = self.lib.get_num_positions
        self.get_num_positions.restype = ctypes.c_ulonglong
        self.get_num_positions.argtypes = [
            ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ctypes.c_ulonglong
        ]

        self.get_nnue_data = self.lib.get_nnue_data
        self.get_nnue_data.restype = ctypes.c_ulonglong
        self.get_nnue_data.argtypes = [
            ctypes.c_char_p,
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
        ]

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
        print(f'Found {self.n_games} games with {self.n_pos} positions')

        # Train-test split
        if test_size * self.n_games > 1:
            self.train_indices, self.test_indices = train_test_split(
                np.arange(self.n_games), test_size=test_size
            )
        else:
            self.train_indices = np.arange(self.n_games)
            self.test_indices = np.array([], dtype=np.int64)
        self.train_games = len(self.train_indices)
        self.test_games = len(self.test_indices)
        self.train = True

    def set_train(self) -> None:
        """Set the dataset to training mode."""
        self.train = True

    def set_test(self) -> None:
        """Set the dataset to testing mode."""
        self.train = False

    def num_train_games(self) -> int:
        """Return the number of training games."""
        return self.train_games

    def num_test_games(self) -> int:
        """Return the number of testing games."""
        return self.test_games

    def __len__(self) -> int:
        return self.train_games if self.train else self.test_games

    def __getitem__(self, index) -> BatchedData:
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
        n_pos = self.get_nnue_data(
            self.fname,
            self.indices,
            games,
            len(games),
            hash(games.tobytes()),
            self.prob_skip,
            w_idx,
            b_idx,
            w_cols,
            b_cols,
            scores,
            results,
            buckets,
        )
        # Build reduced arrays
        scores_array, results_array, buckets_array = (
            torch.tensor(scores[:n_pos], dtype=torch.float32),
            torch.tensor(results[:n_pos], dtype=torch.float32) / 2 + 0.5,
            torch.tensor(buckets[:n_pos], dtype=torch.long),
        )
        # Build the embedding tensors
        w_offset, b_offset, w_cols, b_cols = (
            torch.LongTensor(np.array(w_idx[:n_pos], dtype=np.int_)),
            torch.LongTensor(np.array(b_idx[:n_pos], dtype=np.int_)),
            torch.LongTensor(np.array(w_cols[:w_idx[n_pos]], dtype=np.int_)),
            torch.LongTensor(np.array(b_cols[:b_idx[n_pos]], dtype=np.int_)),
        )
        return BatchedData(
            w_offset,
            w_cols,
            b_offset,
            b_cols,
            scores_array,
            results_array,
            buckets_array,
        )


class BatchedDataLoader:
    """Batched data loader for PawnDataset."""

    def __init__(self, dataset: PawnDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = BatchSampler(
            RandomSampler(dataset), batch_size=batch_size, drop_last=True
        )

    def __iter__(self):
        return self.sampler.__iter__()

    def load(self, indices: List[int]) -> BatchedData:
        """Load a batch of data given a list of indices."""
        return self.dataset[indices]

    def set_train(self) -> None:
        """Set the dataset to training mode."""
        self.dataset.set_train()

    def set_test(self) -> None:
        """Set the dataset to testing mode."""
        self.dataset.set_test()


class TrainingSession:
    """Training session for the NNUE model."""

    def __init__(
        self, pawn_path: str, dataset_path: str, options: Options
    ) -> None:
        self.options = options
        torch.set_num_threads(1)
        print(f"Using {options.device} device")

        # Define model
        if options.load is None:
            self.model = NNUE()
            self.model = self.model.to(options.device)
        else:
            self.model = torch.load(options.load, map_location=options.device)

        # Build dataset and dataloaders
        self.dataset = PawnDataset(
            pawn_path, dataset_path, options.random_skip, options.test_size
        )
        self.dataloader = BatchedDataLoader(
            self.dataset, options.batch_size
        )

        # Set up other parameters
        self.train_batches, self.test_batches = (
            self.dataset.num_train_games() // options.batch_size,
            self.dataset.num_test_games() // options.batch_size,
        )

        # Optimiser
        self.optimiser = torch.optim.Adam(
            self.model.parameters(), lr=options.lr
        )

        # Set up LR scheduler
        self.scheduler = None
        if options.lr_scheduler == 'step':
            self.scheduler = StepLR(
                self.optimiser,
                step_size=options.lr_step_size,
                gamma=options.lr_step_gamma,
            )
        elif options.lr_scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimiser,
                T_max=options.epochs,
                eta_min=options.lr_cosine_final * options.lr,
            )

    def train(self) -> None:
        """Train the model."""

        # Create output directory and history files
        os.makedirs(self.options.output_dir, exist_ok=True)
        with open('train.hist', 'w', encoding='utf-8') as train_file, \
             open('test.hist', 'w', encoding='utf-8') as test_file:

            # Main training loop
            for epoch in range(self.options.epochs):
                self._train_epoch(epoch, train_file)
                if self.dataset.num_test_games() > 0:
                    self._test_epoch(epoch, test_file)

                # Step the LR scheduler (if any)
                if self.scheduler is not None:
                    self.scheduler.step()

                # Save latest model
                base_path = os.path.join(self.options.output_dir, "model")
                torch.save(self.model, f"{base_path}-latest")

                # Save model every save_interval epochs
                if (epoch + 1) % self.options.save_interval == 0:
                    torch.save(self.model, f"{base_path}-epoch{epoch + 1}")

        # Save final model
        torch.save(self.model, os.path.join(self.options.output_dir, "model"))

    def _train_epoch(self, epoch: int, output_file: TextIO) -> None:
        """Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        output_file : TextIO
            The file to write training history to.
        """

        # Set model and dataset to training mode
        self.model.train()
        self.dataloader.set_train()

        # Build progress bar and parallel map
        pbar = tqdm(
            self.dataloader,
            desc=f'Epoch {epoch + 1}',
            total=self.train_batches,
            unit='batch',
        )
        pmap = ParallelMap(
            self.dataloader.load,
            self.dataloader,
            num_threads=self.options.num_threads,
        )

        # Loop over training batches
        with pbar, pmap:
            for batch, data in enumerate(pmap):
                data = data.to(self.options.device)

                # Compute prediction error
                pred = self.model(*data.model_inputs())
                loss = self.model.loss(pred, *data.loss_targets())

                # Backpropagation
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                # Update progress
                info = self.model.info()
                info_str = f', {info}' if info is not None else ''
                output_file.write(f'{epoch}\t{batch}\t{loss.item()}\n')
                output_file.flush()
                pbar.set_postfix_str(f'Loss: {loss.item():>6.4e}{info_str}')
                pbar.update()

                if (batch + 1) % 100 == 0:
                    base_path = os.path.join(self.options.output_dir, "model")
                    torch.save(self.model, f"{base_path}-latest-batch")

    def _test_epoch(self, epoch, output_file) -> None:
        """Evaluate the model on the test dataset.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        output_file : TextIO
            The file to write test history to.
        """
        # Set eval mode and create parallel map
        self.model.eval()
        self.dataloader.set_test()
        pmap = ParallelMap(
            self.dataloader.load,
            self.dataloader,
            num_threads=self.options.num_threads,
        )

        # Loop over test batches
        vals = []
        with torch.no_grad(), pmap:
            for data in pmap:
                data = data.to(self.options.device)
                pred = self.model(*data.model_inputs())
                vals.append(self.model.loss(pred, *data.loss_targets()).item())
        test_loss = sum(vals) / len(vals)

        # Update progress
        output_file.write(f'{epoch}\t{test_loss}\n')
        output_file.flush()
        info = self.model.info()
        info_str = f', {info}' if info is not None else ''
        print(f'Epoch {epoch + 1}: Test loss: {test_loss:>6.4e}{info_str}')


def main() -> None:
    """Entry point for the training script."""
    parser = ArgumentParser(prog='NNUE trainer for pawn')
    parser.add_argument('dataset', help='Dataset file', type=str)
    parser.add_argument('pawn', help='Shared library of pawn', type=str)
    parser.add_arguments(Options, dest='options')
    args = parser.parse_args()
    session = TrainingSession(args.pawn, args.dataset, args.options)
    session.train()


if __name__ == '__main__':
    main()
