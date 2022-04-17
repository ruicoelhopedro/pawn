#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include <iomanip>


namespace Evaluation
{
    class Attacks
    {
        Bitboard m_total;
        Bitboard m_attacks[NUM_PIECE_TYPES];

    public:
        template<PieceType PIECE>
        void push(Bitboard attacks)
        {
            m_attacks[PIECE] |= attacks;
            m_total |= attacks;
        }

        template <PieceType PIECE>
        Bitboard get() const { return m_attacks[PIECE]; }

        template<PieceType PIECE>
        Bitboard get_less_valuable() const
        {
            if constexpr (PIECE == PAWN)
                return Bitboard();
            else if (PIECE == KNIGHT || PIECE == BISHOP)
                return m_attacks[PAWN];
            else if (PIECE == ROOK)
                return get_less_valuable<BISHOP>() | m_attacks[KNIGHT] | m_attacks[BISHOP];
            else if (PIECE == QUEEN)
                return get_less_valuable<  ROOK>() | m_attacks[  ROOK];
            else if (PIECE == KING)
                return get_less_valuable< QUEEN>() | m_attacks[ QUEEN];
        }

        inline Bitboard get() const { return m_total; }
    };


    struct PawnStructure
    {
        Bitboard attacks;
        Bitboard span;
        Bitboard outposts;
        Bitboard open_files;
        Bitboard passed;
    };


    struct EvalFields
    {
        MixedScore material;
        MixedScore placement;
        MixedScore space;
        std::array<MixedScore, NUM_PIECE_TYPES> pieces;
    };


    struct EvalData
    {
        EvalData(const Board& board);

        Attacks attacks[NUM_COLORS];
        PawnStructure pawns[NUM_COLORS];
        Bitboard king_zone[NUM_COLORS];
        Bitboard king_attackers[NUM_COLORS];
        EvalFields fields[NUM_COLORS];
    };


    Score evaluation(const Board& board, EvalData& data);


    void eval_table(const Board& board, EvalData& data, Score score);


    template<bool SKIP>
    class Term
    {
        MixedScore m_white;
        MixedScore m_black;

    public:
        Term(MixedScore white, MixedScore black)
            : m_white(white), m_black(black)
        {}

        template<bool S>
        friend std::ostream& operator<<(std::ostream& out, const Term<S>& term);
    };


    template<bool SKIP>
    std::ostream& operator<<(std::ostream& out, const Term<SKIP>& term)
    {
        out << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
        if (SKIP)
        {
            out << "  --     --   |   --     --   | ";
        }
        else
        {
            out << std::setw(5) << term.m_white.middlegame() / 100.0 << "  ";
            out << std::setw(5) << term.m_white.endgame()    / 100.0 << "  ";
            out << "| ";
            out << std::setw(5) << term.m_black.middlegame() / 100.0 << "  ";
            out << std::setw(5) << term.m_black.endgame()    / 100.0 << "  ";
            out << "| ";
        }
        out << std::setw(5) << (term.m_white.middlegame() - term.m_black.middlegame()) / 100.0;
        out << "  ";
        out << std::setw(5) << (term.m_white.endgame() - term.m_black.endgame()) / 100.0;
        out << "  ";
        return out;
    }
}

template<bool OUTPUT>
Score evaluate(const Position& pos)
{
    const Board& board = pos.board();
    Evaluation::EvalData data(board);

    Score score = Evaluation::evaluation(board, data);

    if (OUTPUT)
        Evaluation::eval_table(board, data, score);

    return score;
}