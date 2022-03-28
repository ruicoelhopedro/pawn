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


class MoveList
{
    Move* m_moves;
    Move* m_end;

public:
    MoveList()
        : m_moves(nullptr), m_end(nullptr)
    {
    }

    MoveList(Move* start)
        : m_moves(start), m_end(start)
    {
    }

    MoveList(Move* start, Move* end)
        : m_moves(start), m_end(end)
    {
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

    int length() const
    {
        return m_end - m_moves;
    }

    void clear()
    {
        m_end = m_moves;
    }

    bool contains(Move move) const
    {
        for (Move* i = m_moves; i < m_end; i++)
            if (*i == move)
                return true;

        return false;
    }

    void pop_first()
    {
        m_moves++;
    }

    // Iterators
    Move* begin() { return m_moves; }
    Move* end()   { return m_end; }
    const Move* begin() const { return m_moves; }
    const Move* end()   const { return m_end; }
    const Move* cbegin() const { return m_moves; }
    const Move* cend()   const { return m_end; }

    // IO operators
    friend std::ostream& operator<<(std::ostream& out, const MoveList& list);
};


class MoveStack
{
    Move* m_moves;
    int m_depth;
    int m_current;

public:
    MoveStack(int depth = 1)
        : m_moves(new Move[NUM_MAX_MOVES * depth]), m_depth(depth), m_current(0)
    {
    }

    virtual ~MoveStack()
    {
        delete[] m_moves;
    }

    MoveStack(const MoveStack& other)
        : m_moves(new Move[NUM_MAX_MOVES * other.m_depth]), m_depth(other.m_depth), m_current(other.m_current)
    {
        for (int i = 0; i < NUM_MAX_MOVES * other.m_depth; i++)
            m_moves[i] = other.m_moves[i];
    }

    MoveStack(MoveStack&& other) noexcept
        : m_moves(other.m_moves), m_depth(other.m_depth), m_current(other.m_current)
    {
        other.m_moves = nullptr;
    }

    MoveStack& operator=(const MoveStack& other)
    {
        if (&other != this)
        {
            if (m_depth != other.m_depth)
            {
                m_depth = other.m_depth;
                Move* tmp = new Move[NUM_MAX_MOVES * m_depth];
                delete[] m_moves;
                m_moves = tmp;
            }

            m_current = other.m_current;
            for (int i = 0; i < NUM_MAX_MOVES * m_depth; i++)
                m_moves[i] = other.m_moves[i];
        }
        return *this;
    }

    MoveStack& operator=(MoveStack&& other) noexcept
    {
        std::swap(m_moves, other.m_moves);
        std::swap(m_depth, other.m_depth);
        std::swap(m_current, other.m_current);
        return *this;
    }

    void reset_pos()
    {
        m_current = 0;
    }

    MoveList list() const
    {
        return MoveList(m_moves + m_current * NUM_MAX_MOVES);
    }

    MoveList list(int pos) const
    {
        return MoveList(m_moves + pos * NUM_MAX_MOVES);
    }

    MoveStack& operator++()
    {
        m_current++;
        return *this;
    }

    MoveStack& operator--()
    {
        m_current--;
        return *this;
    }
};
