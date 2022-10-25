#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include <cassert>
#include <fstream>
#include <string>

namespace PSQT
{
    PSQT* psqt_data;


    PSQT::PSQT(std::string file)
    {
        std::ifstream input(file, std::ios_base::binary);
        assert(input.is_open() && "Failed to open PSQ input file!");
        assert(input.read(reinterpret_cast<char *>(m_weights), sizeof(m_weights)) && "Failed to read PSQ data!");
    }


    Score PSQT::get(PieceType piece, Square square, Turn turn, Square king_sq, Turn king_turn) const
    {
        // No PSQ data for the king
        if (piece == KING)
            return 0;

        // Vertical mirror for black
        if (turn == BLACK)
        {
            king_turn = ~king_turn;
            square = vertical_mirror(square);
            king_sq = vertical_mirror(king_sq);
        }

        // Horiziontal mirror if king on the files E to H
        if (file(king_sq) >= 4)
        {
            square = horizontal_mirror(square);
            king_sq = horizontal_mirror(king_sq);
        }

        // Compute corrected king index (since we are mirrored there are only 4 files)
        int king_index = 4 * rank(king_sq) + file(king_sq);

        // Compute PSQ table index
        int index = square
                    + piece      *  NUM_SQUARES
                    + king_index * (NUM_SQUARES * (NUM_PIECE_TYPES - 1))
                    + king_turn  * (NUM_SQUARES * (NUM_PIECE_TYPES - 1) * NUM_SQUARES / 2);

        return m_weights[index];
    }


    void init()
    {
        load(PSQT_Default_File);
    }

    
    void load(std::string file)
    {
        if (psqt_data)
            delete psqt_data;

        psqt_data = new PSQT(file);
    }
}
