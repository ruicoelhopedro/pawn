#include "Move.hpp"
#include <iostream>


std::ostream& operator<<(std::ostream& out, const Move& move)
{
	out << move.to_uci();
	return out;
}


std::ostream& operator<<(std::ostream& out, const MoveList& list)
{
	if (list.lenght() > 0)
	{
		out << list.m_moves[0];
		for (Move* i = (list.m_moves + 1); i < list.m_end; i++)
			out << " " << *i;
	}
	return out;
}