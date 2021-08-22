#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
#include <vector>

TranspositionEntry::TranspositionEntry()
	: m_hash(0), m_depth(0), m_type(EntryType::EXACT), m_score(SCORE_NONE), m_best_move(Move()), m_static_eval(SCORE_NONE)
{}

TranspositionEntry::TranspositionEntry(Hash hash, Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
	: m_depth(depth), m_score(score), m_best_move(best_move), m_type(type), m_static_eval(static_eval),
	  m_hash(hash ^ data_hash(depth, score, best_move, type, static_eval))
{
}


Hash TranspositionEntry::data_hash(Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
{
	return (static_cast<Hash>(depth)              <<  0)
		 | (static_cast<Hash>(score)              <<  8)
		 | (static_cast<Hash>(best_move.to_int()) << 24)
		 | (static_cast<Hash>(type)               << 40)
		 | (static_cast<Hash>(static_eval)        << 48);
}

bool TranspositionEntry::is_empty() const { return m_score == SCORE_NONE; }
EntryType TranspositionEntry::type() const { return m_type; }
Depth TranspositionEntry::depth() const { return m_depth; }
Score TranspositionEntry::score() const { return m_score; }
Score TranspositionEntry::static_eval() const { return m_static_eval; }
Hash TranspositionEntry::hash() const { return m_hash ^ data_hash(m_depth, m_score, m_best_move, m_type, m_static_eval); }
Move TranspositionEntry::hash_move() const { return m_best_move; }
void TranspositionEntry::reset() { m_depth = 0; }



PerftEntry::PerftEntry()
	: m_hash(0), m_depth(0), m_nodes(-1)
{}

PerftEntry::PerftEntry(Hash hash, Depth depth, int n_nodes)
	: m_hash(hash), m_depth(depth), m_nodes(std::max(0, n_nodes))
{}

bool PerftEntry::is_empty() const { return m_nodes < 0; }
Depth PerftEntry::depth() const { return m_depth; }
Hash PerftEntry::hash() const { return m_hash; }
int PerftEntry::n_nodes() const { return m_nodes; }
void PerftEntry::reset() { m_nodes = -1; }



TranspositionTable<TranspositionEntry> ttable;
TranspositionTable<PerftEntry> perft_table;