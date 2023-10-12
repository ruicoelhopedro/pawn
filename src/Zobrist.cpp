#include "Zobrist.hpp"
#include "Types.hpp"
#include "Move.hpp"

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
        PseudoRandom rnd(54651);

        for (int i = 0; i < NUM_PIECE_TYPES; i++)
            for (int j = 0; j < NUM_COLORS; j++)
                for (int k = 0; k < NUM_SQUARES; k++)
                    randoms::rnd_piece_turn_square[i][j][k] = rnd.next();

        randoms::rnd_black_move = rnd.next();

        for (int i = 0; i < NUM_CASTLE_SIDES; i++)
            for (int j = 0; j < NUM_COLORS; j++)
                randoms::rnd_castle_side_turn[i][j] = rnd.next();

        for (int i = 0; i < 8; i++)
            randoms::rnd_ep_file[i] = rnd.next();
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


    Hash get_move_hash(Move move)
    {
        // Based on a simple linear congruential generator
        // These values have been tested for collisions for all possible Move values
        return 0x89b4fa525 * move.to_int() + 0xe3b2eb29df24cba7;
    }
}
