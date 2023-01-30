import os
import sys
import argparse
import subprocess
from multiprocessing.dummy import Pool as ThreadPool


def get_seed(filename):
    tmp = filename.split('-')[1]
    return int(tmp.split('.')[0])


class DataGenerator:

    def __init__(self, pawn_path, games_per_batch, depth, output_dir):
        self.n_games = 0
        self.n_batches = 0
        self.pawn_path = pawn_path
        self.games_per_batch = games_per_batch
        self.depth = depth
        self.output_dir = output_dir

    def report(self):
        sys.stdout.write(f'\r{self.n_games} games completed ({self.n_batches} batches)')

    def run_batch(self, seed):
        ofile = os.path.join(self.output_dir, f'output-{seed}.dat')
        args = [self.pawn_path, 'play_games',
                'runs_per_fen', str(self.games_per_batch),
                'depth', str(self.depth),
                'seed', str(seed),
                'output_file', ofile]
        result = subprocess.run(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        if result.returncode == 0:
            self.n_games += self.games_per_batch
            self.n_batches += 1
    


def main(pawn_path, games_per_batch, depth, output_dir, num_threads, num_batches, init_seed):
    # Build command line generator
    gen = DataGenerator(pawn_path, games_per_batch, depth, output_dir)
    # Try to fetch last used seed from the output directory
    seeds = [get_seed(a) for a in os.listdir(output_dir) if a.startswith('output-') and a.endswith('.dat')]
    seed = max(seeds) + 1 if len(seeds) > 0 else init_seed
    # Run the batches
    pool = ThreadPool(num_threads)
    batches = range(seed, seed + num_batches, 1)
    datagen = lambda seed: gen.run_batch(seed)
    result = pool.imap_unordered(datagen, batches, chunksize=1)
    for _ in result:
        gen.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Data generator for pawn')
    parser.add_argument('pawn_path')
    parser.add_argument('output_dir')
    parser.add_argument('--games_per_batch', default=1000, type=int)
    parser.add_argument('--depth', default=9, type=int)
    parser.add_argument('--num_threads', default=1, type=int)
    parser.add_argument('--num_batches', default=1000000, type=int)
    parser.add_argument('--init_seed', default=0, type=int)

    args = parser.parse_args()
    main(args.pawn_path, args.games_per_batch, args.depth, args.output_dir, args.num_threads, args.num_batches, args.init_seed)
