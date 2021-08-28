#pragma once
#include "Types.hpp"

namespace Zobrist
{
    namespace randoms
    {
        extern Hash rnd_piece_turn_square[NUM_PIECE_TYPES][NUM_COLORS][NUM_SQUARES];
        extern Hash rnd_black_move;
        extern Hash rnd_castle_side_turn[NUM_COLORS][NUM_CASTLE_SIDES];
        extern Hash rnd_ep_file[8];
    }

    void build_rnd_hashes();
    Hash get_rnd_number();

    Hash get_piece_turn_square(PieceType piece, Turn turn, Square square);
    Hash get_black_move();
    Hash get_castle_side_turn(CastleSide side, Turn turn);
    Hash get_ep_file(int file);
}
