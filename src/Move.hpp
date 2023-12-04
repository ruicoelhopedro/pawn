#pragma once
#include "Types.hpp"
#include <iostream>
#include <string>
#include <cassert>
#include <utility>

class Move
{
    uint16_t m_move;

public:
    constexpr Move()
        : m_move(0)
    {}
    Move(Square from, Square to)
        : m_move(from + (to << 6))
    {}
    Move(Square from, Square to, MoveType mt)
        : m_move(from + (to << 6) + (mt << 12))
    {}
    Move(Hash hash)
        : m_move((hash & 0b111111111111) | ((hash & 0b11000000000000) << 3))
    {}

    static Move from_int(uint16_t number)
    {
        Move move;
        move.m_move = number;
        return move;
    }

    constexpr Square from() const { return m_move & 0b111111; }
    constexpr Square to() const { return (m_move & 0b111111000000) >> 6; }
    constexpr bool is_capture() const { return move_type() & CAPTURE; }
    constexpr bool is_ep_capture() const { return move_type() == EP_CAPTURE; }
    constexpr bool is_double_pawn_push() const { return move_type() == DOUBLE_PAWN_PUSH; }
    constexpr bool is_promotion() const { return move_type() & KNIGHT_PROMO; }
    constexpr bool is_castle() const { return (move_type() == KING_CASTLE) || (move_type() == QUEEN_CASTLE); }
    constexpr PieceType promo_piece() const { return static_cast<PieceType>((move_type() & 0b0011) + 1); }
    constexpr MoveType move_type() const { return static_cast<MoveType>(m_move >> 12); }
    constexpr Hash hash() const { return (m_move & 0b111111111111) | ((m_move & 0b11000000000000000) >> 3); }
    constexpr Hash to_int() const { return m_move; }

    std::string to_uci() const
    {
        if (from() == to())
            return "0000";

        if (is_promotion())
        {
            // Promotion
            PieceType piece = promo_piece();
            char promo_code = piece == KNIGHT ? 'n'
                            : piece == BISHOP ? 'b'
                            : piece == ROOK   ? 'r'
                            :                   'q';
            return get_square(from()) + get_square(to()) + promo_code;
        }
        else
        {
            // Regular move
            return get_square(from()) + get_square(to());
        }
    }

    constexpr bool operator==(const Move& other) const { return m_move == other.m_move; }
    constexpr bool operator!=(const Move& other) const { return m_move != other.m_move; }

    // IO operators
    friend std::ostream& operator<<(std::ostream& out, const Move& move);
};


constexpr Move MOVE_NULL = Move();


class MoveList : public StaticVector<Move, NUM_MAX_MOVES>
{
public:
    template<bool IS_CAPTURE>
    void push_promotions(Square from, Square to)
    {
        if (IS_CAPTURE)
        {
            push(from, to, QUEEN_PROMO_CAPTURE);
            push(from, to, KNIGHT_PROMO_CAPTURE);
            push(from, to, ROOK_PROMO_CAPTURE);
            push(from, to, BISHOP_PROMO_CAPTURE);
        }
        else
        {
            push(from, to, QUEEN_PROMO);
            push(from, to, KNIGHT_PROMO);
            push(from, to, ROOK_PROMO);
            push(from, to, BISHOP_PROMO);
        }
    }   
};
