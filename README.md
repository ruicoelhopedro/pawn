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
The ratings are computed with `ordo` (anchored at zero for the first commit and with errors relative to pool average) using the `noob_3moves` book under LTC conditions (single-thread with TC 60+0.6 and 64MB hash).

| Date     | Commit  |  Elo   | Error(+/-) |
|----------|---------|--------|------------|
| 23/10/06 | [`b079af8`](https://github.com/ruicoelhopedro/pawn/commit/b079af8fdb84dc061dbbed21ff29f60e7312bdaa) | 1619.7 |       17.9 |
| 23/08/31 | [`5a5ea8a`](https://github.com/ruicoelhopedro/pawn/commit/5a5ea8a6d999208831f9d12521a0d379450611ae) | 1556.2 |       17.8 |
| 23/06/30 | [`b68c1df`](https://github.com/ruicoelhopedro/pawn/commit/b68c1dfbae0a5346401dff5655a87ff5f7555862) | 1549.4 |       17.8 |
| 23/04/24 | [`c777eef`](https://github.com/ruicoelhopedro/pawn/commit/c777eef3af07889de88d51d830eb1e6f3c8423ba) | 1508.0 |       16.5 |
| 23/03/12 | [1.0](https://github.com/ruicoelhopedro/pawn/tree/v1.0)                                             | 1387.7 |       16.1 |
| 23/02/11 | [`527fe63`](https://github.com/ruicoelhopedro/pawn/commit/527fe63b24f58b1230278c1f4dfc4541d0472f99) | 1369.7 |       15.5 |
| 23/01/31 | [`d7b26dc`](https://github.com/ruicoelhopedro/pawn/commit/d7b26dcab6ba7ff8bc451c17423a0cec31dd8d6a) | 1350.6 |       15.3 |
| 22/12/20 | [`ecf549f`](https://github.com/ruicoelhopedro/pawn/commit/ecf549f4d3c988fcf44b34402cdb534c71881056) | 1161.2 |       13.8 |
| 22/11/18 | [`c22a7e5`](https://github.com/ruicoelhopedro/pawn/commit/c22a7e526826ad10565cbddb49a907351f8e27ba) | 1147.1 |       14.1 |
| 22/10/06 | [`567797f`](https://github.com/ruicoelhopedro/pawn/commit/567797f0fbe1df386444df12e07462dc2305bb60) | 1026.7 |       14.2 |
| 22/09/23 | [`132140b`](https://github.com/ruicoelhopedro/pawn/commit/132140b23e018c82259061b46d8b4569ac429666) | 1016.9 |       17.0 |
| 22/08/30 | [`25607d9`](https://github.com/ruicoelhopedro/pawn/commit/25607d9b1d1357164d49438f95a81a78109930e2) |  982.8 |       16.8 |
| 22/07/30 | [`638dc4c`](https://github.com/ruicoelhopedro/pawn/commit/638dc4cfe2b2b9d15832d5be6811d17989535185) |  924.1 |       17.1 |
| 22/06/27 | [`069e93a`](https://github.com/ruicoelhopedro/pawn/commit/069e93aedd915f2826b06e838e727c915249591d) |  825.1 |       17.6 |
| 22/05/27 | [`78c2f15`](https://github.com/ruicoelhopedro/pawn/commit/78c2f15ab191cb5ffc9e40244e6ec9bc807a0622) |  631.8 |       19.3 |
| 22/04/27 | [`5fd6e1d`](https://github.com/ruicoelhopedro/pawn/commit/5fd6e1d74b4efdaf7c86044db5f6fd5c52dbcbb5) |  382.4 |       21.6 |
| 22/03/18 | [`fa8e828`](https://github.com/ruicoelhopedro/pawn/commit/fa8e8281278eaad998446b8137db4a1708b05411) |  346.3 |       21.4 |
| 22/02/28 | [`0a131bd`](https://github.com/ruicoelhopedro/pawn/commit/0a131bdb01c8d5cbfa1f68de349e5ca4bcb9dec8) |  325.0 |       22.0 |
| 21/10/29 | [`61edb2a`](https://github.com/ruicoelhopedro/pawn/commit/61edb2a9417ed3b03ddfb8a667c883d55c44036d) |  313.4 |       21.9 |
| 21/09/28 | [`cadf61b`](https://github.com/ruicoelhopedro/pawn/commit/cadf61b0049c37b06c14ef64b43b0eaa8cea0610) |  194.0 |       23.9 |
| 21/08/31 | [`056c448`](https://github.com/ruicoelhopedro/pawn/commit/056c44850f6d74a993b1c1eee10a2809cd99c889) |    0.0 |       30.3 |


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

- #### NNUE_File
  Path to the NNUE net file to use (empty by default). If empty falls back to the embedded network file.

- #### SyzygyPath
  Path to the directory containing the Syzygy endgame tablebases.

- #### SyzygyProbeDepth
  Minimum depth to probe the endgame tablebases during the search.

- #### SyzygyProbeLimit
  Maximum number of pieces to allow probing the table.

  
Furthermore, the following non-standard commands are available:
- `board` - show a representation of the current board;
- `eval` - print some of the evaluation terms;
- `test` - test the move generation, transposition tables, move orderers and legality checks of the engine;
- `bench` - search a set of positions to obtain a signature. Optionally, the following syntax is available: `bench depth threads hash`. By default, calling `bench` is equivalent to `bench 13 1 16`.
- `go perft depth` - do the `perft` node count for the current position at depth `depth`.

## Main Features

### Syzygy endgame tablebases
The Syzygy endgame tablebases support uses [Fathom](https://github.com/jdart1/Fathom) for probing the WDL and DTZ tables.
Similarly to the original implementation in Stockfish, `pawn` has two modes of operation:
- If the root position **is not** on the tablebase, `pawn` will probe the WDL tables during the search (only after a move that resets the 50-move rule counter);
- If the root position **is** on the tablebase, then only the moves that preserve the root's WDL score are searched.
  The engine will continue to search on these moves, but the reported score is derived from the tablebase (unless a mate score is found).
  To avoid wasting moves in long conversions or possibly drawing by repetition, DTZ tables are used to make progress whenever two-fold repetitions may arise.
  When using `go searchmoves`, the move selection is ignored, but TB scores are still returned.


### Board representation
- Bitboard representation (with GCC builtin bitscan and popcount) 
- Magic numbers for slider move generation
- Staged move generation for captures and quiet moves
### Evaluation
- Efficiently updatable neural network (NNUE):
  - Combined PSQ and layer contributions
  - HalfKP_hm feature set
  - Two perspective networks for symmetric evaluations
  - Tapered score from middlegame and endgame network outputs
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
