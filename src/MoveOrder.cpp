#include "MoveOrder.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Hash.hpp"
#include "PieceSquareTables.hpp"
#include <iostream>


Histories::Histories()
{
    clear();
}


void Histories::clear()
{
    for (int i = 0; i < NUM_COLORS; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            for (int k = 0; k < NUM_SQUARES; k++)
                m_butterfly[i][j][k] = 0;

    for (int i = 0; i < NUM_KILLERS; i++)
        for (int j = 0; j < NUM_MAX_DEPTH; j++)
            m_killers[i][j] = MOVE_NULL;
            
    for (int i = 0; i < NUM_SQUARES; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            for (int k = 0; k < NUM_PIECE_TYPES; k++)
                for (int l = 0; l < NUM_SQUARES; l++)
                    m_continuation[i][j][k][l] = 0;
}


void Histories::add_bonus(Move move, Turn turn, PieceType piece, Move prev_move, int bonus)
{
    saturate_add<15000>(m_butterfly[turn][move.from()][move.to()], bonus);
    saturate_add<30000>(m_continuation[prev_move.from()][prev_move.to()][piece][move.to()], bonus);
}


void Histories::bestmove(Move move, Move prev_move, Turn turn, Depth depth, Depth ply, PieceType piece)
{
    add_bonus(move, turn, piece, prev_move, hist_bonus(depth));

    // Exit if killer already in the list
    if (is_killer(move, ply))
        return;

    // Right-shift killers and add new one
    for (int i = NUM_KILLERS - 1; i > 0; i--)
        m_killers[i][ply] = m_killers[i - 1][ply];
    m_killers[0][ply] = move;
}


bool Histories::is_killer(Move move, Depth ply) const
{
    for (int i = 0; i < NUM_KILLERS; i++)
        if (m_killers[i][ply] == move)
            return true;
    return false;
}


int Histories::butterfly_score(Move move, Turn turn) const
{
    return m_butterfly[turn][move.from()][move.to()];
}

int Histories::continuation_score(Move move, PieceType piece, Move prev_move) const
{
    return m_continuation[prev_move.from()][prev_move.to()][piece][move.to()];
}

Move Histories::get_killer(int index, Depth ply) const
{
    return m_killers[index][ply];
}


MoveOrder::MoveOrder(Position& pos, Depth ply, Depth depth, Move hash_move, const Histories& histories, bool quiescence)
    : m_position(pos), m_ply(ply), m_depth(depth), m_hash_move(hash_move), m_histories(histories),
      m_prev_move(pos.last_move()), m_quiescence(quiescence), m_stage(MoveStage::HASH),
      m_killer(MOVE_NULL)
{
}


constexpr MoveStage operator++(MoveStage& current)
{
    current = static_cast<MoveStage>(static_cast<int>(current) + 1);
    return current;
}


bool MoveOrder::hash_move(Move& move)
{
    move = m_hash_move;
    return m_position.board().legal(m_hash_move);
}



int MoveOrder::capture_score(Move move) const
{
    // MVV
    PieceType to = move.is_ep_capture() ? PAWN : m_position.board().get_piece_at(move.to());
    return piece_value_mg[to];
}



int MoveOrder::quiet_score(Move move) const
{
    // Quiets are scored based on:
    // 1. Butterfly histories
    // 2. Piece type-destination histories
    PieceType piece = m_position.board().get_piece_at(move.from());
    return m_histories.butterfly_score(move, m_position.get_turn())
         + 4 * m_histories.continuation_score(move, piece, m_prev_move);
}


Move MoveOrder::next_move()
{
    Move move;
    while (true)
    {
        if (m_stage == MoveStage::HASH)
        {
            ++m_stage;
            if (hash_move(move))
                return move;
        }
        else if (m_stage == MoveStage::CAPTURES_INIT)
        {
            ++m_stage;
            m_captures = m_position.move_list();
            m_position.board().generate_moves(m_captures, MoveGenType::CAPTURES);
            partial_sort<true>(m_captures, 0);
            m_curr = m_captures.begin();
            m_bad_captures = m_captures.end();
        }
        else if (m_stage == MoveStage::CAPTURES)
        {
            while (next(move, m_bad_captures))
                if (move != m_hash_move)
                {
                    // At this stage only return good captures
                    if (m_position.board().see(move, std::max(-250, -50 * (m_depth - 1))) >= 0)
                        return move;

                    // Move bad captures to be tried later
                    std::swap(*(--m_curr), *(--m_bad_captures));
                }
            ++m_stage;
        }
        else if (m_stage == MoveStage::CAPTURES_END)
        {
            // Stop providing moves in non-check quiescence
            if (m_quiescence && !m_position.in_check())
                return MOVE_NULL;
            ++m_stage;
        }
        else if (m_stage == MoveStage::KILLERS)
        {
            ++m_stage;
            m_killer = MOVE_NULL;
            for (int i = 0; i < NUM_KILLERS; i++)
            {
                Move candidate = m_histories.get_killer(i, m_ply);
                if (candidate != m_hash_move &&
                    m_position.board().legal(candidate))
                {
                    m_killer = candidate;
                    return m_killer;
                }
            }
        }
        else if (m_stage == MoveStage::QUIET_INIT)
        {
            ++m_stage;
            m_moves = MoveList(m_captures.end());
            m_position.board().generate_moves(m_moves, MoveGenType::QUIETS);
            partial_sort<false>(m_moves, -1000 * m_depth);
            m_curr = m_moves.begin();
        }
        else if (m_stage == MoveStage::QUIET)
        {
            while (next(move, m_moves.end()))
                if (move != m_hash_move && move != m_killer)
                    return move;
            ++m_stage;
        }
        else if (m_stage == MoveStage::BAD_CAPTURES_INIT)
        {
            ++m_stage;
            m_curr = m_bad_captures;
        }
        else if (m_stage == MoveStage::BAD_CAPTURES)
        {
            while (next(move, m_captures.end()))
                if (move != m_hash_move)
                    return move;
            ++m_stage;
        }
        else
        {
            return MOVE_NULL;
        }
    }
}
