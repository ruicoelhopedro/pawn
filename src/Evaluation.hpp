#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include <iomanip>
#include <array>


class Thread;

class Position;

namespace Evaluation
{

    Score evaluation(const Board& board);


    void eval_table(const Board& board);


    class Term
    {
        Score m_score;

    public:
        Term(Score score)
            : m_score(score)
        {}

        static inline double adjust(Score s) { return double(s) / ScoreToCp; }

        friend std::ostream& operator<<(std::ostream& out, const Term& term);
    };


    inline std::ostream& operator<<(std::ostream& out, const Term& term)
    {
        out << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
        out << std::setw(6) << term.adjust(term.m_score);
        return out;
    }
}