#include "MoveOrder.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "Move.hpp"
#include "Transpositions.hpp"
#include "PieceSquareTables.hpp"
#include <iostream>


Histories::Histories()
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


void Histories::add_bonus(Move move, Turn turn, PieceType piece, int bonus)
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
	m_low_ply_history[ply - 1][piece][move.to()] += value;
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
	return ply < NUM_LOW_PLY ? m_low_ply_history[ply - 1][piece][move.to()] : 0;
}


MoveOrder::MoveOrder(Position& pos, Depth depth, Move hash_move, const Histories& histories, Move prev_move, bool quiescence)
	: m_position(pos),  m_stage(MoveStage::HASH), m_quiescence(quiescence), 
	  m_hash_move(hash_move), m_histories(histories), m_depth(depth), 
	  m_countermove(MOVE_NULL), m_killer(MOVE_NULL), m_prev_move(prev_move)   
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


bool MoveOrder::capture_move(Move& move)
{
	Score curr_score = -SCORE_INFINITE;
	Move* curr_move = nullptr;

	// Find move with greater score
	auto list_move = m_moves.begin();
	while (list_move != m_moves.end())
	{
		Score score = capture_score(*list_move);
		if (score > curr_score && *list_move != m_hash_move)
		{
			curr_score = score;
			curr_move = list_move;
		}

		list_move++;
	}

	if (curr_move != nullptr)
	{
		// Pop the move and return
		move = *curr_move;
		m_moves.pop(curr_move);
		return true;
	}

	return false;
}


bool MoveOrder::quiet_move(Move& move)
{
	Score curr_score = -SCORE_INFINITE;
	Move* curr_move = nullptr;

	// Find move with greater score
	auto list_move = m_moves.begin();
	while (list_move != m_moves.end())
	{
		Score score = quiet_score(*list_move);
		if (score > curr_score && *list_move != m_hash_move)
		{
			curr_score = score;
			curr_move = list_move;
		}

		list_move++;
	}

	if (curr_move != nullptr)
	{
		// Pop the move and return
		move = *curr_move;
		m_moves.pop(curr_move);
		return true;
	}

	return false;
	//auto list_move = m_quiets.begin();
	//while (list_move != m_quiets.end())
	//{
	//	if (*list_move != m_hash_move)
	//	{
	//		// Pop the move and return
	//		move = *list_move;
	//		m_quiets.pop(list_move);
	//		return true;
	//	}

	//	list_move++;
	//}

	//return false;
}


Score MoveOrder::capture_score(Move move) const
{
	// MVV-LVA
	constexpr Score piece_score[] = { 10, 30, 31, 50, 90, 1000 };
	Piece from = m_position.board().get_piece_at(move.from());
	Piece to = (move.is_ep_capture()) ? PAWN : m_position.board().get_piece_at(move.to());
	return piece_score[to] - piece_score[from];
}



Score MoveOrder::quiet_score(Move move) const
{
	// Quiets are scored based on:
	// 1. Butterfly histories
	// 2. Piece type-destination histories
	// 3. Low ply histories (based on node counts)
	// 4. PSQT difference
	Turn turn = m_position.board().turn();
	auto piece = static_cast<PieceType>(m_position.board().get_piece_at(move.from()));
	return m_histories.butterfly_score(move, m_position.get_turn())
		 + m_histories.piece_type_score(move, piece)
		 + m_histories.low_ply_score(move, piece, m_position.ply())
		 + (piece_square(piece, move.to(), turn) - piece_square(piece, move.from(), turn)).tapered(m_position.board().phase());
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
			m_moves = m_position.move_list();
			m_position.board().generate_moves(m_moves, MoveGenType::CAPTURES);
			//sort_moves<true>(threshold_moves<true>(m_moves, -SCORE_INFINITE));
			++m_stage;
		}
		else if (m_stage == MoveStage::CAPTURES)
		{
			if (capture_move(move))
				if (move != m_hash_move)
					return move;
			++m_stage;
		}
		else if (m_stage == MoveStage::COUNTERMOVES)
		{
			++m_stage;
			m_countermove = m_histories.countermove(m_prev_move);
			if (!m_quiescence && m_countermove != MOVE_NULL && 
				m_countermove != m_hash_move && 
				m_position.board().legal(m_countermove))
				return m_countermove;
		}
		else if (m_stage == MoveStage::KILLERS)
		{
			++m_stage;
			m_killer = MOVE_NULL;
			if (!m_quiescence)
				for (int i = 0; i < NUM_KILLERS; i++)
				{
					Move killer_candidate = m_histories.get_killer(i, m_position.ply());
					if (killer_candidate != m_hash_move && 
						killer_candidate != m_countermove && 
						m_position.board().legal(killer_candidate))
					{
						m_killer = killer_candidate;
						break;
					}
				}
			if (m_killer != MOVE_NULL)
				return m_killer;
		}
		else if (m_stage == MoveStage::QUIET_INIT)
		{
			m_moves = m_position.move_list();
			if (m_position.is_check() || !m_quiescence)
			{
				m_position.board().generate_moves(m_moves, MoveGenType::QUIETS);
				//sort_moves<false>(threshold_moves<false>(m_moves, 10000));
			}
			++m_stage;
		}
		else if (m_stage == MoveStage::QUIET)
		{
			while (quiet_move(move))
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