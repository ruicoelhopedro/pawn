# pawn
A UCI alpha-beta chess engine, largely inspired by Stockfish.

As with most UCI chess engines, `pawn` should be used with a compatible graphical user interface (such as [CuteChess](https://github.com/cutechess/cutechess)).

The engine uses a hybrid evaluation function combining both handcrafted terms (such as material, mobility and king safety) and an efficiently updatable neural network for positional scores (derived from a set of trained PSQ tables in earlier versions).
All training data has been generated in self-play at low depth using the tools in the branch [`data_gen`](https://github.com/ruicoelhopedro/pawn/tree/data_gen).
The default network is embedded in the binary file with `incbin`.

## Getting `pawn`
The latest official version can be downloaded in the [Releases](https://github.com/ruicoelhopedro/pawn/releases) tab.
Binaries for both Windows and Linux are available.
For each platform, two 64-bit versions are provided: a generic version for most x86 processors (slower) and one with the instruction set from `haswell` (faster with BMI2, AVX2 and POPCNT, but may not be supported for older CPUs).

For the most up-to-date (and likely stronger) version, it is recommended to compile directly from sources for the target architecture, as described below.

### Compiling from sources
To compile under Linux or Windows with MSYS2, simply call
```
make
```
in the root directory to generate an executable optimised for the building machine.
The resulting binary and object files can be found in the `build` directory.
Verify the signature of the binary by running the `bench` command and checking if the number of nodes searched matches the Bench field of the last commit message. 

To compile generic binaries or to a specific architecture, use
```
make ARCH=target
```
where `target` is the target passed to `-march=`.


## Progress

The progression of `pawn` in self-play since the first public version is illustrated in the table below.
The ratings are computed with `ordo` anchored at zero mean using the `noob_3moves` book and with TC 60+0.6.

| Date     | Commit  |  Elo   | Error(+/-) |
|----------|---------|--------|------------|
| 23/02/11 | [`527fe63`](https://github.com/ruicoelhopedro/pawn/commit/527fe63b24f58b1230278c1f4dfc4541d0472f99) |  620.9 |       25.5 |
| 23/01/31 | [`d7b26dc`](https://github.com/ruicoelhopedro/pawn/commit/d7b26dcab6ba7ff8bc451c17423a0cec31dd8d6a) |  602.4 |       23.6 |
| 22/12/20 | [`ecf549f`](https://github.com/ruicoelhopedro/pawn/commit/ecf549f4d3c988fcf44b34402cdb534c71881056) |  409.5 |       20.2 |
| 22/11/18 | [`c22a7e5`](https://github.com/ruicoelhopedro/pawn/commit/c22a7e526826ad10565cbddb49a907351f8e27ba) |  393.9 |       19.4 |
| 22/10/06 | [`567797f`](https://github.com/ruicoelhopedro/pawn/commit/567797f0fbe1df386444df12e07462dc2305bb60) |  279.7 |       18.7 |
| 22/09/23 | [`132140b`](https://github.com/ruicoelhopedro/pawn/commit/132140b23e018c82259061b46d8b4569ac429666) |  267.0 |       19.8 |
| 22/08/30 | [`25607d9`](https://github.com/ruicoelhopedro/pawn/commit/25607d9b1d1357164d49438f95a81a78109930e2) |  232.9 |       19.1 |
| 22/07/30 | [`638dc4c`](https://github.com/ruicoelhopedro/pawn/commit/638dc4cfe2b2b9d15832d5be6811d17989535185) |  174.2 |       18.4 |
| 22/06/27 | [`069e93a`](https://github.com/ruicoelhopedro/pawn/commit/069e93aedd915f2826b06e838e727c915249591d) |   75.3 |       19.6 |
| 22/05/27 | [`78c2f15`](https://github.com/ruicoelhopedro/pawn/commit/78c2f15ab191cb5ffc9e40244e6ec9bc807a0622) | -118.0 |       19.5 |
| 22/04/27 | [`5fd6e1d`](https://github.com/ruicoelhopedro/pawn/commit/5fd6e1d74b4efdaf7c86044db5f6fd5c52dbcbb5) | -367.4 |       21.1 |
| 22/03/18 | [`fa8e828`](https://github.com/ruicoelhopedro/pawn/commit/fa8e8281278eaad998446b8137db4a1708b05411) | -403.5 |       21.7 |
| 22/02/28 | [`0a131bd`](https://github.com/ruicoelhopedro/pawn/commit/0a131bdb01c8d5cbfa1f68de349e5ca4bcb9dec8) | -424.8 |       21.8 |
| 21/10/29 | [`61edb2a`](https://github.com/ruicoelhopedro/pawn/commit/61edb2a9417ed3b03ddfb8a667c883d55c44036d) | -436.4 |       22.0 |
| 21/09/28 | [`cadf61b`](https://github.com/ruicoelhopedro/pawn/commit/cadf61b0049c37b06c14ef64b43b0eaa8cea0610) | -555.8 |       24.2 |
| 21/08/31 | [`056c448`](https://github.com/ruicoelhopedro/pawn/commit/056c44850f6d74a993b1c1eee10a2809cd99c889) | -749.8 |       30.3 |


## UCI Options
The following UCI options are supported:
- #### Hash
  Size of the Hash Table, in MB (defaults to 16).
  
- #### Threads
  The number of threads to use during search (defaults to 1).
 
- #### MultiPV
  The number of principal variations (PV) to search (defaults to 1). This should be kept at 1 for best performance.
 
- #### Ponder
  Allow the engine to think during the opponent's move (defaults to false). This requires the GUI to send the appropriate `go ponder`.
 
- #### Move Overhead
  Extra time to reserve for each move (defaults to 0). Increase if the engine flags due to communication delays.

- #### Clear Hash
  Button to clear the hash table.

- #### PSQT_File
  Path to the PSQT net file to use (empty by default). If empty falls back to the embedded network file.

  
Furthermore, the following non-standard commands are available:
- `board` - show a representation of the current board;
- `eval` - print some of the evaluation terms;
- `test` - test the move generation, transposition tables, move orderers and legality checks of the engine;
- `bench` - search a set of positions to obtain a signature. Optionally, the following syntax is available: `bench depth threads hash`. By default, calling `bench` is equivalent to `bench 13 1 16`.
- `go perft depth` - do the `perft` node count for the current position at depth `depth`.

## Main Features

### Board representation
- Bitboard representation (with GCC builtin bitscan and popcount) 
- Magic numbers for slider move generation
- Staged move generation for captures and quiet moves
### Evaluation
- Tapered evaluation
- Material imbalance
- Efficiently updatable positional neural network for positional score
- Basic pawn structure
- Mobility
- Per-piece bonuses
- King safety
- Basic threats
- Space
- Basic endgame scaling functions
### Search
- Principal Variation Search in a negamax framework
- Quiescence search with SEE
- MultiPV search
- Transposition Tables
- Aspiration Windows
- Late move reductions
- Null-move pruning
- Singular and check extensions
- Futility pruning
- Mate distance pruning
- Basic Lazy SMP threading
### Move Ordering
- MVV move ordering for captures
- Killer moves
- Quiet move ordering
  - History heuristic with butterfly and 3-ply continuation piece-to boards
