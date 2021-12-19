#include "MoveOrder.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Transpositions.hpp"
#include "PieceSquareTables.hpp"
#include <iostream>
#include <cstring>


HistoryContext::HistoryContext(Histories& hist, Board board, Depth ply, Depth depth, Move prev_move)
    : m_hist(hist), m_board(board), m_ply(ply), m_depth(depth), m_prev_move(prev_move)
{
}


void HistoryContext::add_bonus(Move move, int bonus)
{
    m_hist.add_bonus(move, m_board.turn(), static_cast<PieceType>(m_board.get_piece_at(move.from())), m_ply, bonus);
}


void HistoryContext::fail_high(Move move)
{
    m_hist.fail_high(move, m_prev_move, m_board.turn(), m_depth, m_ply, static_cast<PieceType>(m_board.get_piece_at(move.from())));
}


void HistoryContext::update_low_ply(Move move, int bonus)
{
    m_hist.update_low_ply(move, m_ply, static_cast<PieceType>(m_board.get_piece_at(move.from())), bonus);
}



bool HistoryContext::is_killer(Move move) const
{
    return m_hist.is_killer(move, m_ply);
}


int HistoryContext::butterfly_score(Move move) const
{
    return m_hist.butterfly_score(move, m_board.turn());
}


int HistoryContext::piece_type_score(Move move) const
{
    return m_hist.piece_type_score(move, static_cast<PieceType>(m_board.get_piece_at(move.from())));
}


int HistoryContext::low_ply_score(Move move) const
{
    return m_hist.low_ply_score(move, static_cast<PieceType>(m_board.get_piece_at(move.from())), m_ply);
}


Move HistoryContext::countermove() const
{
    return m_hist.countermove(m_prev_move);
}


Move HistoryContext::get_killer(int index) const
{
    return m_hist.get_killer(index, m_ply);
}



Histories::Histories()
{
    clear();
}


void Histories::add_bonus(Move move, Turn turn, PieceType piece, Depth ply, int bonus)
{
    m_butterfly[turn][move.from()][move.to()] += bonus;
    m_piece_type[piece][move.to()] += bonus;
}


void Histories::fail_high(Move move, Move prev_move, Turn turn, Depth depth, Depth ply, PieceType piece)
{
    m_butterfly[turn][move.from()][move.to()] += depth * depth;
    m_piece_type[piece][move.to()] += depth * depth;
    m_countermoves[prev_move.from()][prev_move.to()] = move;

    // Exit if killer already in the list
    if (is_killer(move, ply))
        return;

    // Right-shift killers and add new one
    for (int i = NUM_KILLERS - 1; i > 0; i--)
        m_killers[i][ply] = m_killers[i - 1][ply];
    m_killers[0][ply] = move;
}


void Histories::update_low_ply(Move move, Depth ply, PieceType piece, int value)
{
    m_low_ply_history[ply][piece][move.to()] += value;
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


int Histories::piece_type_score(Move move, PieceType piece) const
{
    return m_piece_type[piece][move.to()];
}


Move Histories::get_killer(int index, Depth ply) const
{
    return m_killers[index][ply];
}


Move Histories::countermove(Move move) const
{
    return m_countermoves[move.from()][move.to()];
}


int Histories::low_ply_score(Move move, PieceType piece, Depth ply) const
{
    return ply < NUM_LOW_PLY ? m_low_ply_history[ply][piece][move.to()] : 0;
}


void Histories::clear()
{
    for (int i = 0; i < NUM_COLORS; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            for (int k = 0; k < NUM_SQUARES; k++)
                m_butterfly[i][j][k] = 0;
                
    for (int i = 0; i < NUM_PIECE_TYPES; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            m_piece_type[i][j] = 0;

    for (int i = 0; i < NUM_KILLERS; i++)
        for (int j = 0; j < NUM_MAX_DEPTH; j++)
            m_killers[i][j] = MOVE_NULL;

    for (int i = 0; i < NUM_SQUARES; i++)
        for (int j = 0; j < NUM_SQUARES; j++)
            m_countermoves[i][j] = MOVE_NULL;

    for (int i = 0; i < NUM_LOW_PLY; i++)
        for (int j = 0; j < NUM_PIECE_TYPES; j++)
            for (int k = 0; k < NUM_SQUARES; k++)
                m_low_ply_history[i][j][k] = 0;
}



MoveOrder::MoveOrder(Position& pos, Move hash_move, const HistoryContext& histories, bool quiescence)
    : m_position(pos), m_hash_move(hash_move), m_histories(histories), m_quiescence(quiescence), 
      m_stage(MoveStage::HASH), m_countermove(MOVE_NULL), m_killer(MOVE_NULL)
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
    // MVV-LVA
    constexpr int piece_score[] = { 10, 30, 31, 50, 90, 1000 };
    Piece from = m_position.board().get_piece_at(move.from());
    Piece to = (move.is_ep_capture()) ? PAWN : m_position.board().get_piece_at(move.to());
    return piece_score[to] - piece_score[from];
}



int MoveOrder::quiet_score(Move move) const
{
    // Quiets are scored based on:
    // 1. Butterfly histories
    // 2. Piece type-destination histories
    // 3. Low ply histories (based on node counts)
    return m_histories.butterfly_score(move)
         + m_histories.piece_type_score(move)
         + m_histories.low_ply_score(move);
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
            m_moves = m_position.move_list();
            m_position.board().generate_moves(m_moves, MoveGenType::CAPTURES);
        }
        else if (m_stage == MoveStage::CAPTURES)
        {
            while (best_move<true>(move))
                if (move != m_hash_move)
                    return move;
            ++m_stage;
        }
        else if (m_stage == MoveStage::CAPTURES_END)
        {
            // Stop providing moves in non-check quiescence
            if (m_quiescence && !m_position.is_check())
                return MOVE_NULL;
            ++m_stage;
        }
        else if (m_stage == MoveStage::COUNTERMOVES)
        {
            ++m_stage;
            Move candidate = m_histories.countermove();
            if (candidate != m_hash_move &&
                m_position.board().legal(candidate))
            {
                m_countermove = candidate;
                return m_countermove;
            }
        }
        else if (m_stage == MoveStage::KILLERS)
        {
            ++m_stage;
            m_killer = MOVE_NULL;
            for (int i = 0; i < NUM_KILLERS; i++)
            {
                Move candidate = m_histories.get_killer(i);
                if (candidate != m_hash_move &&
                    candidate != m_countermove &&
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
            m_moves = m_position.move_list();
            m_position.board().generate_moves(m_moves, MoveGenType::QUIETS);
        }
        else if (m_stage == MoveStage::QUIET)
        {
            while (best_move<false>(move))
                if (move != m_hash_move && move != m_killer && move != m_countermove)
                    return move;
            ++m_stage;
        }
        else
        {
            return MOVE_NULL;
        }
    }
}
