#include "Zobrist.hpp"
#include "Types.hpp"

namespace Zobrist
{
    namespace randoms
    {
        Hash rnd_piece_turn_square[NUM_PIECE_TYPES][NUM_COLORS][NUM_SQUARES];
        Hash rnd_black_move;
        Hash rnd_castle_side_turn[NUM_COLORS][NUM_CASTLE_SIDES];
        Hash rnd_ep_file[8];
    }


    void build_rnd_hashes()
    {
        for (int i = 0; i < NUM_PIECE_TYPES; i++)
            for (int j = 0; j < NUM_COLORS; j++)
                for (int k = 0; k < NUM_SQUARES; k++)
                    randoms::rnd_piece_turn_square[i][j][k] = get_rnd_number();

        randoms::rnd_black_move = get_rnd_number();

        for (int i = 0; i < NUM_CASTLE_SIDES; i++)
            for (int j = 0; j < NUM_COLORS; j++)
                randoms::rnd_castle_side_turn[i][j] = get_rnd_number();

        for (int i = 0; i < 8; i++)
            randoms::rnd_ep_file[i] = get_rnd_number();
    }


    Hash get_rnd_number()
    {
        Hash u1 = rand() & 0xFFFF;
        Hash u2 = rand() & 0xFFFF;
        Hash u3 = rand() & 0xFFFF;
        Hash u4 = rand() & 0xFFFF;
        return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
    }


    Hash get_piece_turn_square(PieceType piece, Turn turn, Square square)
    {
        return randoms::rnd_piece_turn_square[piece][turn][square];
    }


    Hash get_black_move()
    {
        return randoms::rnd_black_move;
    }


    Hash get_castle_side_turn(CastleSide side, Turn turn)
    {
        return randoms::rnd_castle_side_turn[side][turn];
    }


    Hash get_ep_file(int file)
    {
        return randoms::rnd_ep_file[file];
    }
}
