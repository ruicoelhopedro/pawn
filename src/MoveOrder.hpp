#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Hash.hpp"


enum class MoveStage
{
    HASH,
    CAPTURES_INIT,
    CAPTURES,
    CAPTURES_END,
    KILLERS,
    QUIET_INIT,
    QUIET,
    BAD_CAPTURES_INIT,
    BAD_CAPTURES,
    NO_MOVES
};

constexpr MoveStage operator++(MoveStage& current);


constexpr int NUM_KILLERS = 3;
constexpr int NUM_LOW_PLY = 5;


class Histories
{
    Move m_killers[NUM_KILLERS][NUM_MAX_DEPTH];
    int m_butterfly[NUM_COLORS][NUM_SQUARES][NUM_SQUARES];
    int m_continuation[NUM_SQUARES][NUM_SQUARES][NUM_PIECE_TYPES][NUM_SQUARES];

public:
    Histories();

    void clear();

    void add_bonus(Move move, Turn turn, PieceType piece, Move prev_move, int bonus);
    void bestmove(Move move, Move prev_move, Turn turn, Depth depth, Depth ply, PieceType piece);

    bool is_killer(Move move, Depth ply) const;
    int butterfly_score(Move move, Turn turn) const;
    int continuation_score(Move move, PieceType piece, Move prev_move) const;
    Move get_killer(int index, Depth ply) const;
};


template<int MAX>
void saturate_add(int& entry, int bonus)
{
    entry += bonus - entry * abs(bonus) / MAX;
}


constexpr int hist_bonus(Depth d)
{
    return std::min(2000, 5 * (d + 10) * d);
}


class MoveOrder
{
    Position& m_position;
    Depth m_ply;
    Depth m_depth;
    Move m_hash_move;
    const Histories& m_histories;
    Move m_prev_move;
    bool m_quiescence;
    MoveList m_moves;
    MoveStage m_stage;
    Move m_killer;
    Move* m_curr;
    MoveList m_captures;
    Move* m_bad_captures;

    bool hash_move(Move& move);


    template<bool CAPTURES>
    int move_score(Move move) const
    {
        if (CAPTURES)
            return capture_score(move);
        else
            return quiet_score(move);
    }


    bool next(Move& move, Move* end)
    {
        if (m_curr == end)
            return false;

        move = *(m_curr++);
        return true;
    }


    template<bool CAPTURES>
    void partial_sort(MoveList& list, int threshold)
    {
        // Partial move sorting based on Stockfish's partial insertion sort
        Move* sorted_end = list.begin();
        for (Move* i = list.begin() + 1; i < list.end(); i++)
        {
            // Only sort moves above the threshold
            int i_score = move_score<CAPTURES>(*i);
            if (i_score > threshold)
            {
                // Store a copy of the move to assign later
                Move curr = *i;

                // Little hack: instead of doing the insertion sort from the current position,
                // shortcut the entries below threshold. To do so, the current entry is swapped
                // with the end of the sorted region. The unsorted entry is set right away,
                // while the sorted one is only inserted at the end.
                *i = *(++sorted_end);
                Move* j = sorted_end;

                // Actual insertion sorting
                for (; j > list.begin() && i_score > move_score<CAPTURES>(*(j - 1)); j--)
                    *j = *(j - 1);
                *j = curr;
            }
        }
    }


public:
    MoveOrder(Position& pos, Depth ply, Depth depth, Move hash_move, const Histories& histories, bool quiescence = false);

    Move next_move();

    int capture_score(Move move) const;
    int quiet_score(Move move) const;
};
