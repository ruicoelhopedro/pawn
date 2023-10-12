#include "Types.hpp"
#include "Bitboard.hpp"
#include "Position.hpp"
#include "Evaluation.hpp"
#include "NNUE.hpp"
#include "Thread.hpp"
#include <cassert>
#include <stdlib.h>

namespace Evaluation
{

Score evaluation(const Board& board)
{
    const NNUE::Accumulator& white_acc = board.accumulator(WHITE);
    const NNUE::Accumulator& black_acc = board.accumulator(BLACK);
    MixedScore mixed_result = white_acc.eval() - black_acc.eval();
    Score result = mixed_result.tapered(board.phase());
    return std::clamp(result, -SCORE_MATE_FOUND + 1, SCORE_MATE_FOUND - 1);
}


void eval_table(const Board& board)
{
    // No eval printing when in check
    if (board.checkers())
    {
        std::cout << "No eval: in check" << std::endl;
        return;
    }

    // Get total and partial NNUE scores
    auto white_acc = board.accumulator(WHITE);
    auto black_acc = board.accumulator(BLACK);
    MixedScore mixed_nnue = white_acc.eval() - black_acc.eval();
    MixedScore psqt = white_acc.eval_psq() - black_acc.eval_psq();
    MixedScore positional = mixed_nnue - psqt;
    Score nnue = mixed_nnue.tapered(board.phase());


    // Print board with NNUE-derived piece values
    Board b = board;
    const char* white_piece_map = "PNBRQK  ";
    const char* black_piece_map = "pnbrqk  ";
    std::cout << std::endl;
    std::cout << "NNUE-derived piece values" << std::endl;
    for (int rank = 7; rank >= 0; rank--)
    {
        std::cout << "+---------+---------+---------+---------+---------+---------+---------+---------+" << std::endl;

        // First pass: write piece types
        std::cout << "|";
        for (int file = 0; file < 8; file++)
        {
            Square s = make_square(rank, file);
            PieceType p = board.get_piece_at(s);
            Turn t = board.get_pieces<WHITE>().test(s) ? WHITE : BLACK;
            std::cout << "    " << (t == WHITE ? white_piece_map[p]: black_piece_map[p]) << "    |";
        }
        std::cout << std::endl;

        // Second pass: write piece values
        std::cout << "|";
        for (int file = 0; file < 8; file++)
        {
            Square s = make_square(rank, file);
            PieceType p = board.get_piece_at(s);
            if (p != PIECE_NONE && p != KING)
            {
                // Compute the value of each piece by evaluating the eval difference after removing it
                Turn t = board.get_pieces<WHITE>().test(s) ? WHITE : BLACK;
                b.pop_piece(p, t, s);
                Score value = nnue - evaluation(b);
                b.set_piece(p, t, s);
                std::cout << " "
                          << std::showpoint << std::noshowpos << std::fixed
                          << std::setprecision(2) << std::setw(6)
                          << Term::adjust(value) << "  |";
            }
            else
                std::cout << "         |";
        }
        std::cout << std::endl;
    }
    std::cout << "+---------+---------+---------+---------+---------+---------+---------+---------+" << std::endl;


    // Print NNUE table
    std::cout << ""                                                   << std::endl;
    std::cout << "NNUE Scores"                                        << std::endl;
    std::cout << "--------------------------------------"             << std::endl;
    std::cout << " Term                |    MG      EG  "             << std::endl;
    std::cout << "--------------------------------------"             << std::endl;
    std::cout << " Material   (PSQT)   | " << Term(psqt)       << " " << std::endl;
    std::cout << " Positional (Layers) | " << Term(positional) << " " << std::endl;
    std::cout << "--------------------------------------"             << std::endl;
    std::cout << " Final               | " << Term(mixed_nnue) << " " << std::endl;
    std::cout << "--------------------------------------"             << std::endl;
    std::cout << "Game Phase:       " << int(board.phase()) << std::endl;
    std::cout << "Final evaluation: " <<  100 * nnue / PawnValue.endgame() << " cp (White)" << std::endl;
    std::cout << std::endl;
}
    
}