#include "Fathom/src/tbprobe.h"
#include "syzygy.hpp"
#include "../UCI.hpp"
#include <iostream>
#include <string>

namespace Syzygy
{
    bool Loaded;
    RootPos Root;
    int Cardinality;


    void init()
    {
        Loaded = false;
        Cardinality = 0;
    }


    void load(std::string path)
    {
        if (Loaded)
            clear();

        if (!tb_init(path.c_str()))
        {
            std::cerr << "Failed to load tablebases" << std::endl;
            std::terminate();
        }

        Loaded = true;
        Cardinality = std::min(int(TB_LARGEST), UCI::Options::TB_ProbeLimit);
    }


    void clear()
    {
        tb_free();
        Loaded = false;
    }



    RootPos::RootPos()
        : m_score(SCORE_NONE),
          m_num_moves(0),
          m_num_preserving_moves(0)
    {}

    RootPos::RootPos(Position& pos)
    {
        // Assume the position is not on TB for initialising data
        m_num_moves = 0;
        m_num_preserving_moves = 0;
        m_score = SCORE_NONE;

        // Now check if it actually is in the TB
        if (pos.board().get_pieces().count() <= Cardinality)
        {
            // Check for a successful probe
            auto probe_result = probe_dtz(pos.board());
            WDL root_wdl = probe_result.first;
            if (root_wdl != WDL_NONE)
            {
                // Generate root moves
                Move moves[NUM_MAX_MOVES];
                MoveList root_moves = MoveList(moves);
                pos.board().generate_moves(root_moves, MoveGenType::LEGAL);

                // Evaluate each root move using DTZ tables
                bool success = root_moves.length() > 0;
                for (const Move& move : root_moves)
                {
                    WDL wdl;
                    int dtz;

                    pos.make_move(move);

                    // Score this move. We need some care to ensure optimal play
                    if (pos.is_draw(false))
                    {
                        // Avoid draws by repetitions
                        wdl = WDL_DRAW;
                        dtz = 0;
                    }
                    else
                    {
                        // Use DTZ to measure progress
                        auto pr = probe_dtz(pos.board());
                        wdl = -pr.first;
                        dtz = pos.board().half_move_clock() >= 4 ? pr.second : 0;
                    }

                    pos.unmake_move();

                    // Store this root move
                    Score score = score_from_wdl(wdl, std::min(dtz, NUM_MAX_PLY - 1));
                    success &= (score != -SCORE_NONE);
                    m_moves[m_num_moves++] = RootMove{ score, dtz, move };
                }

                // Sort the list of moves
                std::stable_sort(m_moves, m_moves + m_num_moves, [](RootMove a, RootMove b)
                {
                    return a.tb_score >= b.tb_score;
                });

                // Find the number of moves that exactly preserve the best observed score
                Score best_score = m_moves[0].tb_score;
                for (m_num_preserving_moves = 0; m_num_preserving_moves < m_num_moves; m_num_preserving_moves++)
                    if (m_moves[m_num_preserving_moves].tb_score < best_score)
                        break;

                // Ensure we only assign TB data if all probes have been successful
                if (success)
                    m_score = root_wdl;
            }
        }
    }

    bool RootPos::in_tb() const
    {
        return m_score != SCORE_NONE;
    }

    int RootPos::num_preserving_moves() const
    {
        return m_num_preserving_moves;
    }

    Score RootPos::move_score(Move move) const
    {
        for (int i = 0; i < m_num_moves; i++)
            if (m_moves[i].move == move)
                return m_moves[i].tb_score;
        return -SCORE_NONE;
    }

    Move RootPos::ordered_moves(int idx) const
    {
        return m_moves[idx].move;
    }



    WDL probe_wdl(const Board& board)
    {
        auto result = tb_probe_wdl(
            board.get_pieces<WHITE>().to_uint64(),
            board.get_pieces<BLACK>().to_uint64(),
            (board.get_pieces<WHITE,   KING>() | board.get_pieces<BLACK,   KING>()).to_uint64(),
            (board.get_pieces<WHITE,  QUEEN>() | board.get_pieces<BLACK,  QUEEN>()).to_uint64(),
            (board.get_pieces<WHITE,   ROOK>() | board.get_pieces<BLACK,   ROOK>()).to_uint64(),
            (board.get_pieces<WHITE, BISHOP>() | board.get_pieces<BLACK, BISHOP>()).to_uint64(),
            (board.get_pieces<WHITE, KNIGHT>() | board.get_pieces<BLACK, KNIGHT>()).to_uint64(),
            (board.get_pieces<WHITE,   PAWN>() | board.get_pieces<BLACK,   PAWN>()).to_uint64(),
            0,
            0,
            0,
            board.turn() == WHITE
        );

        // Convert to internal WDL scores
        return result == TB_RESULT_FAILED ? WDL_NONE : fathom_to_wdl(result);
    }



    std::pair<WDL, int> probe_dtz(const Board& board)
    {
        auto result = tb_probe_root(
            board.get_pieces<WHITE>().to_uint64(),
            board.get_pieces<BLACK>().to_uint64(),
            (board.get_pieces<WHITE,   KING>() | board.get_pieces<BLACK,   KING>()).to_uint64(),
            (board.get_pieces<WHITE,  QUEEN>() | board.get_pieces<BLACK,  QUEEN>()).to_uint64(),
            (board.get_pieces<WHITE,   ROOK>() | board.get_pieces<BLACK,   ROOK>()).to_uint64(),
            (board.get_pieces<WHITE, BISHOP>() | board.get_pieces<BLACK, BISHOP>()).to_uint64(),
            (board.get_pieces<WHITE, KNIGHT>() | board.get_pieces<BLACK, KNIGHT>()).to_uint64(),
            (board.get_pieces<WHITE,   PAWN>() | board.get_pieces<BLACK,   PAWN>()).to_uint64(),
            board.half_move_clock(),
            board.can_castle(),
            board.ep_square() == SQUARE_NULL ? 0 : board.ep_square(),
            board.turn() == WHITE,
            nullptr
        );

        // Disambiguate termination and failure results
        WDL wdl = result == TB_RESULT_FAILED    ? WDL_NONE
                : result == TB_RESULT_CHECKMATE ? WDL_LOSS
                : result == TB_RESULT_STALEMATE ? WDL_DRAW
                :                                 fathom_to_wdl(TB_GET_WDL(result));

        // Convert to internal WDL scores
        return std::make_pair(wdl, TB_GET_DTZ(result));
    }



    WDL fathom_to_wdl(unsigned int result)
    {
        return result == TB_WIN          ? WDL_WIN
             : result == TB_CURSED_WIN   ? WDL_CURSED_WIN
             : result == TB_DRAW         ? WDL_DRAW
             : result == TB_BLESSED_LOSS ? WDL_BLESSED_LOSS
             : result == TB_LOSS         ? WDL_LOSS
             :                             WDL_NONE;
    }



    Score score_from_wdl(WDL wdl, Depth ply)
    {
        return wdl == WDL_LOSS         ? -SCORE_MATE_FOUND + 1 + ply
             : wdl == WDL_BLESSED_LOSS ?  SCORE_DRAW - 1
             : wdl == WDL_DRAW         ?  SCORE_DRAW
             : wdl == WDL_CURSED_WIN   ?  SCORE_DRAW + 1
             : wdl == WDL_WIN          ?  SCORE_MATE_FOUND - 1 - ply
             :                           -SCORE_NONE;
    }



    EntryType bound_from_wdl(WDL wdl)
    {
        return wdl == WDL_LOSS         ? EntryType::UPPER_BOUND
             : wdl == WDL_BLESSED_LOSS ? EntryType::LOWER_BOUND
             : wdl == WDL_DRAW         ? EntryType::EXACT
             : wdl == WDL_CURSED_WIN   ? EntryType::UPPER_BOUND
             : wdl == WDL_WIN          ? EntryType::LOWER_BOUND
             :                           EntryType::EMPTY;
    }
}
