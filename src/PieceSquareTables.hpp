#pragma once
#include "incbin/incbin.h"
#include "Types.hpp"

#define PSQT_Default_File "psqt-d4c6ca5d39dd.dat"

using S = MixedScore;

constexpr S   PawnValue(  125,   200);
constexpr S KnightValue(  750,   850);
constexpr S BishopValue(  800,   900);
constexpr S   RookValue( 1200,  1400);
constexpr S  QueenValue( 2500,  2600);
constexpr S   KingValue(10000, 10000); // Needed for SEE and MVV-LVA

constexpr S piece_value[] = { PawnValue,
                              KnightValue,
                              BishopValue,
                              RookValue,
                              QueenValue,
                              KingValue,
                              S(0, 0), // Empty
                              S(0, 0) }; // PIECE_NONE

constexpr Score piece_value_mg[] = { PawnValue.middlegame(),
                                     KnightValue.middlegame(),
                                     BishopValue.middlegame(),
                                     RookValue.middlegame(),
                                     QueenValue.middlegame(),
                                     KingValue.middlegame(),
                                     0,    // Empty
                                     0 };  // PIECE_NONE

constexpr Score piece_value_eg[] = { PawnValue.endgame(),
                                     KnightValue.endgame(),
                                     BishopValue.endgame(),
                                     RookValue.endgame(),
                                     QueenValue.endgame(),
                                     KingValue.endgame(),
                                     0,    // Empty
                                     0 };  // PIECE_NONE


constexpr S imbalance_terms[][NUM_PIECE_TYPES] =
{   // Pawn       Knight     Bishop    Rook         Queen
    { S( 0,  0)                                            }, // Pawn
    { S(14, 10), S(-5, -6)                                 }, // Knight
    { S( 6,  7), S( 1,  2), S( 0, 0)                       }, // Bishop
    { S( 0,  0), S( 7,  4), S( 9, 8), S(-12, -11)          }, // Rook
    { S( 0,  0), S( 9,  9), S(10, 7), S(-11, -13), S(0, 0) }  // Queen
};


namespace PSQT
{
    using FeatureType = int16_t;
    static constexpr std::size_t NUM_FEATURES = 20480;

    extern const FeatureType* psqt_data;

    void init();

    void load(std::string file);

    void clean();
}


Score piece_square(PieceType piece, Square square, Turn turn, Square king_sq, Turn king_turn);