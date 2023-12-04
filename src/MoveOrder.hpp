#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Hash.hpp"
#include <array>
#include <cstring>


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
constexpr int NUM_CONTINUATION = 3;


constexpr int hist_bonus(Depth d)
{
    return std::min(4000, 200 * d);
}


template<int MAX>
void saturate_add(int& entry, int bonus)
{
    entry += bonus - entry * abs(bonus) / MAX;
}


template<int CAP>
class PieceToHistoryStats
{
    int m_data[NUM_PIECE_TYPES][NUM_SQUARES];

public:
    PieceToHistoryStats() { clear(); }
    void clear() { std::memset(m_data, 0, sizeof(m_data)); }
    int get(PieceType p, Square s) const { return m_data[p][s]; }
    void add(PieceType p, Square s, int bonus) { saturate_add<CAP>(m_data[p][s], bonus); }
};


template<int CAP>
class ButterflyHistoryStats
{
    int m_data[NUM_SQUARES][NUM_SQUARES];

public:
    ButterflyHistoryStats() { clear(); }
    void clear() { std::memset(m_data, 0, sizeof(m_data)); }
    int get(Square from, Square to) const { return m_data[from][to]; }
    void add(Square from, Square to, int bonus) { saturate_add<CAP>(m_data[from][to], bonus); }
};


using ButterflyHistory = ButterflyHistoryStats<15000>;
using PieceToHistory = PieceToHistoryStats<30000>;


struct CurrentHistory
{
    Move* killers;
    PieceToHistory* continuation[NUM_CONTINUATION];
    ButterflyHistory* main_history;

    void add_bonus(Move move, PieceType piece, int bonus);
    void bestmove(Move move, PieceType piece, Depth depth);
    bool is_killer(Move move) const;
};


class Histories
{
    PieceToHistory m_continuation[NUM_SQUARES][NUM_SQUARES];
    ButterflyHistory m_main[NUM_COLORS];
    Move m_killers[NUM_MAX_DEPTH][NUM_KILLERS];

public:
    Histories();
    void clear();

    CurrentHistory get(const Position& pos);
};


class MoveOrder
{
    MoveList m_move_list;
    Position& m_position;
    const CurrentHistory& m_history;
    Move* m_curr;
    Move* m_quiet_begin;
    Move* m_bad_captures;
    MoveStage m_stage;
    Move m_hash_move;
    Move m_killer;
    Depth m_depth;
    bool m_quiescence;


    bool hash_move(Move& move);


    template<bool CAPTURES>
    int move_score(Move move) const
    {
        if constexpr (CAPTURES)
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
    void partial_sort(Move* begin, Move* end, int threshold)
    {
        // Partial move sorting based on Stockfish's partial insertion sort
        Move* sorted_end = begin;
        for (Move* i = begin + 1; i < end; i++)
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
                for (; j > begin && i_score > move_score<CAPTURES>(*(j - 1)); j--)
                    *j = *(j - 1);
                *j = curr;
            }
        }
    }


public:
    MoveOrder(Position& pos, Depth depth, Move hash_move, const CurrentHistory& history, bool quiescence = false);

    Move next_move();

    int capture_score(Move move) const;
    int quiet_score(Move move) const;
};
