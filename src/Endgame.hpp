#pragma once
#include "Types.hpp"
#include "Position.hpp"
#include <functional>


using EvalFunc = Score(*)(const Position&);


namespace Endgame
{
    enum Corner
    {
        WHITE_SQUARES,
        BLACK_SQUARES,
        ANY_CORNER
    };

    EvalFunc specialised_eval(const Position& pos);


    Score Drawing(const Position& pos);

    Score KPK(const Position& pos);
    Score KQRxK(const Position& pos);
    Score KQKP(const Position& pos);
    Score KRKP(const Position& pos);

    template <Corner CORNER, Score BONUS, Turn WINNING>
    Score KingToCorner(const Position& pos);
}