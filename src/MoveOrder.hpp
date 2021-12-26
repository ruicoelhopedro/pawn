#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Transpositions.hpp"


enum class MoveStage
{
    HASH,
    CAPTURES_INIT,
    CAPTURES,
    CAPTURES_END,
    COUNTERMOVES,
    KILLERS,
    QUIET_INIT,
    QUIET,
    NO_MOVES
};

constexpr MoveStage operator++(MoveStage& current);


constexpr int NUM_KILLERS = 3;
constexpr int NUM_LOW_PLY = 5;


class Histories
{
    Move m_killers[NUM_KILLERS][NUM_MAX_DEPTH];
    int m_butterfly[NUM_COLORS][NUM_SQUARES][NUM_SQUARES];
    int m_piece_type[NUM_PIECE_TYPES][NUM_SQUARES];
    Move m_countermoves[NUM_SQUARES][NUM_SQUARES];
    int m_low_ply_history[NUM_LOW_PLY][NUM_PIECE_TYPES][NUM_SQUARES];

public:
    Histories();

    void add_bonus(Move move, Turn turn, PieceType piece, Depth ply, int bonus);
    void fail_high(Move move, Move prev_move, Turn turn, Depth depth, Depth ply, PieceType piece);
    void update_low_ply(Move move, Depth ply, PieceType piece, int value);

    bool is_killer(Move move, Depth ply) const;
    int butterfly_score(Move move, Turn turn) const;
    int piece_type_score(Move move, PieceType piece) const;
    int low_ply_score(Move move, PieceType piece, Depth ply) const;
    Move countermove(Move move) const;
    Move get_killer(int index, Depth ply) const;

    void clear();
};


class HistoryContext
{
    Histories& m_hist;
    Board m_board;
    Depth m_ply;
    Depth m_depth;
    Move m_prev_move;

public:
    HistoryContext(Histories& hist, Board board, Depth ply, Depth depth, Move prev_move);

    void add_bonus(Move move, int bonus);
    void fail_high(Move move);
    void update_low_ply(Move move, int bonus);

    bool is_killer(Move move) const;
    int butterfly_score(Move move) const;
    int piece_type_score(Move move) const;
    int low_ply_score(Move move) const;
    Move countermove() const;
    Move get_killer(int index) const;
};


class MoveOrder
{
    Position& m_position;
    Move m_hash_move;
    const HistoryContext& m_histories;
    bool m_quiescence;
    MoveList m_moves;
    MoveStage m_stage;
    Move m_countermove;
    Move m_killer;
    Move* m_curr;
    int m_last_score;

    bool hash_move(Move& move);


    template<bool CAPTURES>
    int move_score(Move move) const
    {
        if (CAPTURES)
            return capture_score(move);
        else
            return quiet_score(move);
    }
    
    
    template<bool CAPTURES>
    bool best_move(Move& move)
    {
        // Do we have any move?
        if (m_moves.begin() == m_moves.end())
            return false;

        auto list_move = m_moves.begin();
        int curr_score = move_score<CAPTURES>(*list_move);
        Move* curr_move = list_move;

        // Find move with greater score
        list_move++;
        while (list_move != m_moves.end())
        {
            int score = move_score<CAPTURES>(*list_move);
            if (score > curr_score)
            {
                curr_score = score;
                curr_move = list_move;
            }

            list_move++;
        }

        m_last_score = curr_score;
        
        // Pop the move and return
        move = *curr_move;
        m_moves.pop(curr_move);
        return true;
    }


    bool next_move(Move& move)
    {
        // Do we have any move?
        if (m_moves.begin() == m_moves.end())
            return false;

        move = *m_curr;
        return (m_curr++) != m_moves.end();
    }


    template<bool CAPTURES>
    MoveList threshold_moves(MoveList& list, int threshold)
    {
        Move* pos = list.begin();
        for (auto list_move = list.begin(); list_move != list.end(); list_move++)
        {
            if (move_score<CAPTURES>(*list_move) > threshold)
            {
                if (pos != list_move)
                    std::swap(*pos, *list_move);
                pos++;
            }
        }

        return MoveList(list.begin(), pos);
    }


    template<bool CAPTURES>
    void sort_moves(MoveList list) const
    {
        for (auto move = list.begin(); move != list.end(); move++)
        {
            Move* best_move = list.begin();
            int best_score = move_score<CAPTURES>(*best_move);
            for (auto other = best_move + 1; other != list.end(); other++)
            {
                int score = move_score<CAPTURES>(*other);
                if (score > best_score)
                {
                    best_move = other;
                    best_score = score;
                }
            }
            if (best_move != move)
                std::swap(*best_move, *move);
        }
    }


public:
    MoveOrder(Position& pos, Move hash_move, const HistoryContext& histories, bool quiescence = false);

    Move next_move();

    int capture_score(Move move) const;
    int quiet_score(Move move) const;
};
