#include "MoveOrder.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Hash.hpp"
#include "NNUE.hpp"
#include <iostream>


void CurrentHistory::add_bonus(Move move, PieceType piece, int bonus)
{
    main_history->add(move.from(), move.to(), bonus);
    for (std::size_t i = 0; i < NUM_CONTINUATION; i++)
        continuation[i]->add(piece, move.to(), bonus);
}


void CurrentHistory::bestmove(Move move, PieceType piece, Depth depth)
{
    add_bonus(move, piece, hist_bonus(depth));

    // Exit if killer already in the list
    if (is_killer(move))
        return;

    // Right-shift killers and add new one
    for (int i = NUM_KILLERS - 1; i > 0; i--)
        killers[i] = killers[i - 1];
    killers[0] = move;
}


bool CurrentHistory::is_killer(Move move) const
{
    for (int i = 0; i < NUM_KILLERS; i++)
        if (killers[i] == move)
            return true;
    return false;
}



Histories::Histories()
{
    clear();
}


void Histories::clear()
{
    for (int i = 0; i < NUM_SQUARES; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            m_continuation[i][j].clear();

    for (int i = 0; i < NUM_COLORS; i++)
        m_main[i].clear();

    for (int i = 0; i < NUM_MAX_DEPTH; i++)
        for (int j = 0; j < NUM_KILLERS; j++)
            m_killers[i][j] = MOVE_NULL;
}


CurrentHistory Histories::get(const Position& pos)
{
    CurrentHistory history;
    history.killers = m_killers[pos.ply()];
    history.main_history = &m_main[pos.get_turn()];
    for (std::size_t i = 0; i < NUM_CONTINUATION; i++)
    {
        Move m = pos.last_move(2 * i);
        history.continuation[i] = &m_continuation[m.from()][m.to()];
    }
    return history;
}



MoveOrder::MoveOrder(Position& pos, Depth depth, Move hash_move, const CurrentHistory& history, bool quiescence)
    : m_position(pos), m_history(history), m_stage(MoveStage::HASH),
      m_hash_move(hash_move), m_killer(MOVE_NULL), m_depth(depth),
      m_quiescence(quiescence)
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
    PieceType piece = m_position.board().get_piece_at(move.from());
    return     m_history.main_history->get(move.from(), move.to())
         + 2 * m_history.continuation[0]->get(piece, move.to())
         +     m_history.continuation[1]->get(piece, move.to())
         +     m_history.continuation[2]->get(piece, move.to());
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
            m_position.board().generate_moves(m_move_list, MoveGenType::CAPTURES);
            partial_sort<true>(m_move_list.begin(), m_move_list.end(), 0);
            m_curr = m_move_list.begin();
            m_bad_captures = m_move_list.end();
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
                Move candidate = m_history.killers[i];
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
            m_quiet_begin = m_move_list.end();
            m_position.board().generate_moves(m_move_list, MoveGenType::QUIETS);
            partial_sort<false>(m_quiet_begin, m_move_list.end(), -1000 * m_depth);
            m_curr = m_quiet_begin;
        }
        else if (m_stage == MoveStage::QUIET)
        {
            while (next(move, m_move_list.end()))
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
            while (next(move, m_quiet_begin))
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
