#pragma once
#include "Types.hpp"
#include "Bitboard.hpp"
#include "Position.hpp"


class Attacks
{
    Bitboard m_attacks[NUM_PIECE_TYPES];
    Bitboard m_double_attacks[NUM_PIECE_TYPES];

public:
    Attacks();

    template<PieceType PIECE>
    void push(Bitboard attacks)
    {
        m_double_attacks[PIECE] |= (m_attacks[PIECE] & attacks);
        m_attacks[PIECE] |= attacks;
    }

    template<PieceType PIECE>
    Bitboard get() const
    {
        return m_attacks[PIECE];
    }

    Bitboard get() const
    {
        return m_attacks[PAWN] | m_attacks[KNIGHT] | m_attacks[BISHOP]
             | m_attacks[ROOK] | m_attacks[ QUEEN] | m_attacks[  KING];
    }

    int count(Square sq) const
    {
        int n = 0;
        for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        {
            n += m_attacks[piece].test(sq);
            n += m_double_attacks[piece].test(sq);
        }
        return n;
    }

    template<PieceType PIECE>
    Bitboard get_double() const
    {
        return m_double_attacks[PIECE];
    }

    Bitboard get_double() const
    {
        return m_double_attacks[PAWN] | m_double_attacks[KNIGHT] | m_double_attacks[BISHOP]
             | m_double_attacks[ROOK] | m_double_attacks[ QUEEN] | m_double_attacks[  KING];
    }
};


struct PawnStructure
{
    Bitboard attacks;
    Bitboard attackable;
    Bitboard blocked;
    Bitboard passed;
    Bitboard isolated;
    Bitboard doubled;
    Bitboard outposts;
    Bitboard backward;
    Bitboard open_files;
    Bitboard levers;
    Bitboard candidate_passer;
    Bitboard behind;
    int islands;
};


struct EvalFields
{
    MixedScore material{ 0, 0 };
    MixedScore placement{ 0, 0 };
    MixedScore pawns{ 0, 0 };
    MixedScore mobility{ 0, 0 };
    MixedScore king_safety{ 0, 0 };
};


struct EvalData
{
    Attacks attacks[NUM_COLORS];
    PawnStructure pawns[NUM_COLORS];
    Bitboard king_zone[NUM_COLORS];
    Bitboard king_attackers[NUM_COLORS];
    EvalFields fields[NUM_COLORS];
};

struct Term
{
    MixedScore m_white;
    MixedScore m_black;
    bool m_skip;

    Term(MixedScore white, MixedScore black, bool skip = false)
        : m_white(white), m_black(black), m_skip(skip)
    {
    }
};

std::ostream& operator<<(std::ostream& stream, MixedScore score);
std::ostream& operator<<(std::ostream& stream, Term score);

enum EvalType
{
    OUTPUT,
    NO_OUTPUT
};

Score evaluation(const Position& position, bool output = false);

void write_field(std::ostream& stream, MixedScore white, MixedScore black);