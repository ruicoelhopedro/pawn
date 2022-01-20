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
    constexpr Piece promo_piece() const { return (move_type() & 0b0011) + 1; }
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
            std::string promo_code;
            Piece piece = promo_piece();
            if (piece == 1)
                promo_code = "n";
            else if (piece == 2)
                promo_code = "b";
            else if (piece == 3)
                promo_code = "r";
            else if (piece == 4)
                promo_code = "q";
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


class MoveList
{
    Move m_moves[NUM_MAX_MOVES];
    Move* m_start;
    Move* m_end;

public:
    MoveList()
        : m_start(m_moves), m_end(m_moves)
    {
    }

    MoveList(Move* start)
        : m_start(start), m_end(start)
    {
    }

    MoveList(Move* start, Move* end)
        : m_start(start), m_end(end)
    {
    }

    MoveList(const MoveList& other)
    {
        m_end = m_moves;
        for (const Move* m = other.m_moves; m != other.m_end; m++)
            *(m_end++) = *m;
        m_start = m_moves + (other.m_start - other.m_moves);
    }

    MoveList operator=(const MoveList& other)
    {
        MoveList list(other);
        return list;
    }

    void push(Move move)
    {
        *(m_end++) = move;
    }

    void push(Square from, Square to)
    {
        *(m_end++) = Move(from, to);
    }

    void push(Square from, Square to, MoveType mt)
    {
        *(m_end++) = Move(from, to, mt);
    }

    template<bool IS_CAPTURE>
    void push_promotions(Square from, Square to)
    {
        if (IS_CAPTURE)
        {
            *(m_end++) = Move(from, to, QUEEN_PROMO_CAPTURE);
            *(m_end++) = Move(from, to, KNIGHT_PROMO_CAPTURE);
            *(m_end++) = Move(from, to, ROOK_PROMO_CAPTURE);
            *(m_end++) = Move(from, to, BISHOP_PROMO_CAPTURE);
        }
        else
        {
            *(m_end++) = Move(from, to, QUEEN_PROMO);
            *(m_end++) = Move(from, to, KNIGHT_PROMO);
            *(m_end++) = Move(from, to, ROOK_PROMO);
            *(m_end++) = Move(from, to, BISHOP_PROMO);
        }
    }

    void pop()
    {
        m_end--;
    }

    void pop(Move* move)
    {
        *move = *(--m_end);
    }

    int lenght() const
    {
        return m_end - m_start;
    }

    void clear()
    {
        m_end = m_start = m_moves;
    }

    bool contains(Move move) const
    {
        for (Move* i = m_start; i < m_end; i++)
            if (*i == move)
                return true;

        return false;
    }

    void pop_first()
    {
        m_start++;
    }

    // Iterators
    Move* begin() { return m_start; }
    Move* end()   { return m_end; }
    const Move* cbegin() const { return m_start; }
    const Move* cend()   const { return m_end; }

    // IO operators
    friend std::ostream& operator<<(std::ostream& out, const MoveList& list);
};
