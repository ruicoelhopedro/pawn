#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include <iomanip>
#include <array>


class Thread;

namespace Evaluation
{
    class Attacks
    {
        Bitboard m_total;
        Bitboard m_double;
        Bitboard m_attacks[NUM_PIECE_TYPES];

    public:
        template<PieceType PIECE>
        void push(Bitboard attacks)
        {
            m_double |= attacks & m_total;
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
        inline Bitboard get_double() const { return m_double; }
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
        MixedScore imbalance;
        MixedScore placement;
        MixedScore space;
        MixedScore passed;
        MixedScore threats;
        MixedScore scale;
        std::array<MixedScore, NUM_PIECE_TYPES> pieces;
    };


    struct EvalData
    {
        EvalData(const Board& board);

        Attacks attacks[NUM_COLORS];
        PawnStructure pawns[NUM_COLORS];
        Bitboard king_zone[NUM_COLORS];
        int n_king_attacks[NUM_COLORS];
        int king_attack_weight[NUM_COLORS];
        Bitboard king_attackers[NUM_COLORS];
        EvalFields fields[NUM_COLORS];
        Bitboard mobility_area[NUM_COLORS];
    };


    class MaterialEntry
    {
        Hash m_hash;
        MixedScore m_imbalance;

    public:
        inline MaterialEntry()
            : m_hash(0),
              m_imbalance(0, 0)
        {}

        inline bool query(Age age, Hash hash, MaterialEntry** entry)
        {
            (void)age;
            *entry = this;
            return hash == m_hash;
        }

        void store(Age age, Hash hash, const Board& board);

        inline bool empty() const { return m_hash == 0; }

        inline Hash hash() const { return m_hash; }
        inline MixedScore imbalance() const { return m_imbalance; }

        friend MaterialEntry material_eval(const Board& board);
    };

    MaterialEntry material_eval(const Board& board);


    MaterialEntry* probe_material(const Board& board, HashTable<MaterialEntry>& table);


    Score evaluation(const Board& board, EvalData& data, Thread& thread);


    void eval_table(const Board& board, EvalData& data);


    class Term
    {
        MixedScore m_score;

    public:
        Term(MixedScore score)
            : m_score(score)
        {}

        static inline double adjust(Score s) { return double(s) / PawnValue.endgame(); }

        friend std::ostream& operator<<(std::ostream& out, const Term& term);
    };


    inline std::ostream& operator<<(std::ostream& out, const Term& term)
    {
        out << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
        out << std::setw(6) << term.adjust(term.m_score.middlegame());
        out << "  ";
        out << std::setw(6) << term.adjust(term.m_score.endgame());
        return out;
    }
}