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

    std::string to_uci() const;

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

    MoveList(const MoveList& list)
    {
        m_start = m_moves + (list.m_start - list.m_moves);
        m_end = m_start;
        for (Move* m = list.m_start; m != list.m_end; m++)
            *(m_end++) = *m;
    }

    MoveList& operator=(const MoveList& list)
    {
        m_start = m_moves + (list.m_start - list.m_moves);
        m_end = m_start;
        for (Move* m = list.m_start; m != list.m_end; m++)
            *(m_end++) = *m;
        return *this;
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
        return m_end - m_start;
    }

    void clear()
    {
        m_start = m_moves;
        m_end = m_start;
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
    Move* begin() const { return m_start; }
    Move* end()   const { return m_end; }
    const Move* cbegin() const { return m_start; }
    const Move* cend()   const { return m_end; }

    Move& operator[](int i) { return *(m_start + i); }
    const Move& operator[](int i) const { return *(m_start + i); }

    // IO operators
    friend std::ostream& operator<<(std::ostream& out, const MoveList& list);
};


constexpr int PV_LENGTH = NUM_MAX_MOVES;
constexpr int TOTAL_PV_LENGTH = (PV_LENGTH * PV_LENGTH + PV_LENGTH) / 2;

struct PvContainer
{
    Move pv[TOTAL_PV_LENGTH];
    MoveList prev_pv;
};

