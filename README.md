# pawn
A UCI alpha-beta chess engine, largely inspired by Stockfish.

As with most UCI chess engines, `pawn` should be used with a compatible graphical user interface (such as [CuteChess](https://github.com/cutechess/cutechess)).

## UCI Options
The following UCI options are supported:
- #### Hash
  Size of the Hash Table, in MB (defaults to 16).
  
- #### Threads
  Number of threads to use during search (defaults to 1).
 
- #### MultiPV
  Number of principal variations (PV) to search (defaults to 1). This should be kept at 1 for best performance.
 
- #### Ponder
  Allow the engine to think during the opponent's move (defaults to false). This requires the GUI to send the appropriate `go ponder`.
  
Furthermore, the following non-standard commands are available:
- `board` - show representation of the current board;
- `eval` - print some of the evaluation terms;
- `test` - test the move generation, transposition tables, move orderers and legality checks of the engine;
- `go perft depth` - do the `perft` node count for the current position at depth `depth`.

## Main Features

### Board representation
- Bitboard representation (with GCC builtin bitscan and popcount) 
- Magic numbers for slider move generation
- Staged move generation for captures and quiet moves
### Evaluation
- Tapered evaluation
- Material and Piece-square tables (incrementally updated)
- Basic pawn structure
- Mobility
- Per-piece bonuses
- King safety
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
- MVV-LVA move ordering for captures
- Killer moves and countermoves
- Quiet move ordering
  - History heuristic with butterfly and piece-to boards
  - Piece-square difference
  - Low-ply histories based on nodes searched

## Compiling from sources
To compile with `g++`, simply call
```
make
```
in the root directory. The resulting binary and object files can be found in the `build` directory.
