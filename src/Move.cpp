#include "Move.hpp"
#include <iostream>


std::string Move::to_uci() const
{
    if (from() == to())
        return "0000";

    if (is_promotion())
    {
        // Promotion
        Piece piece = promo_piece();
        char promo_code = piece == 1 ? 'n'
                        : piece == 2 ? 'b'
                        : piece == 3 ? 'r'
                        : piece == 4 ? 'q'
                        : 'x';
        return get_square(from()) + get_square(to()) + promo_code;
    }
    else
    {
        // Regular move
        return get_square(from()) + get_square(to());
    }
}


std::ostream& operator<<(std::ostream& out, const Move& move)
{
    // From
    out << static_cast<char>('a' + file(move.from()));
    out << static_cast<char>('1' + rank(move.from()));
    // To
    out << static_cast<char>('a' + file(move.to()));
    out << static_cast<char>('1' + rank(move.to()));
    if (move.is_promotion())
    {
        Piece piece = move.promo_piece();
        char promo_code = piece == 1 ? 'n'
                        : piece == 2 ? 'b'
                        : piece == 3 ? 'r'
                        : piece == 4 ? 'q'
                        : 'x';
        out << promo_code;
    }
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
