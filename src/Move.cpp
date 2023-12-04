#include "Move.hpp"
#include <iostream>


std::ostream& operator<<(std::ostream& out, const Move& move)
{
    out << move.to_uci();
    return out;
}
