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


std::size_t eval_bucket(const Board& board) { return (board.get_pieces().count() - 1) / 8; }


Score evaluation(const Board& board)
{
    const Turn stm = board.turn();
    Score result = NNUE::Accumulator::eval(board.accumulator( stm),
                                           board.accumulator(~stm),
                                           eval_bucket(board)) * turn_to_color(stm);
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
    const Turn stm = board.turn();
    Score nnue = evaluation(board);

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
    std::cout << " Bucket   |  PSQT   Layers  |  Total  "             << std::endl;
    std::cout << "--------------------------------------"             << std::endl;
    for (std::size_t bucket = 0; bucket < NNUE::NUM_BUCKETS; bucket++)
    {
        Score s = NNUE::Accumulator::eval(board.accumulator(stm), board.accumulator(~stm), bucket);
        Score psqt = NNUE::Accumulator::eval_psq(board.accumulator(stm), board.accumulator(~stm), bucket);
        Score layers = s - psqt;
        std::cout << " "
                  << int(bucket)
                  << (bucket == eval_bucket(board) ? " (used)" : "       ")
                  << " |"
                  << Term(psqt)
                  << "   "
                  << Term(layers)
                  << "  | "
                  << Term(s)
                  << std::endl;
    }
    std::cout << "--------------------------------------"             << std::endl;
    std::cout << "Final evaluation: " <<  100 * nnue / PawnValue.endgame() << " cp (White)" << std::endl;
    std::cout << std::endl;
}
    
}