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

    void add_bonus(Move move, Turn turn, PieceType piece, int bonus);
    void fail_high(Move move, Move prev_move, Turn turn, Depth depth, Depth ply, PieceType piece);
    void update_low_ply(Move move, Depth ply, PieceType piece, int value);

    bool is_killer(Move move, Depth ply) const;
    int butterfly_score(Move move, Turn turn) const;
    int piece_type_score(Move move, PieceType piece) const;
    int low_ply_score(Move move, PieceType piece, Depth ply) const;
    Move countermove(Move move) const;
    Move get_killer(int index, Depth ply) const;
};


class MoveOrder
{
    bool m_quiescence;
    MoveList m_moves;
    Move m_prev_move;
    Position& m_position;
    MoveStage m_stage;
    Move m_hash_move;
    Move m_countermove;
    Move m_killer;
    Depth m_depth;
    const Histories& m_histories;

    bool hash_move(Move& move);
    bool capture_move(Move& move);
    bool quiet_move(Move& move);
    Score capture_score(Move move) const;
    Score quiet_score(Move move) const;

    template<bool CAPTURES>
    Score move_score(Move move) const
    {
        if (CAPTURES)
        {
            // MVV-LVA
            constexpr Score piece_score[] = { 10, 30, 31, 50, 90, 1000 };
            Piece from = m_position.board().get_piece_at(move.from());
            Piece to = (move.is_ep_capture()) ? PAWN : m_position.board().get_piece_at(move.to());
            return piece_score[to] - piece_score[from];
        }
        else
        {
            return m_histories.butterfly_score(move, m_position.get_turn())
                + m_histories.piece_type_score(move, static_cast<PieceType>(m_position.board().get_piece_at(move.from())));
            //// Promotions go first
            //if (move.is_promotion())
            //	return piece_value[move.promo_piece()] + MixedScore(SCORE_INFINITE, SCORE_INFINITE);

            //// Castling goes second
            //if (move.is_castle())
            //	return MixedScore(SCORE_INFINITE, SCORE_INFINITE);

            //// Compute static evaluation difference
            //Turn turn = m_position.board().turn();
            //Piece piece = m_position.board().get_piece_at(move.from());
            //return piece_square(piece, move.to(), turn) - piece_square(piece, move.from(), turn);
        }
    }

    template<bool CAPTURES>
    MoveList threshold_moves(MoveList& list, Score threshold)
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
            Score best_score = move_score<CAPTURES>(*best_move);
            for (auto other = best_move + 1; other != list.end(); other++)
            {
                Score score = move_score<CAPTURES>(*other);
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
    MoveOrder(Position& pos, Depth depth, Move hash_move, const Histories& histories, Move prev_move, bool quiescence = false);

    Move next_move();
};
