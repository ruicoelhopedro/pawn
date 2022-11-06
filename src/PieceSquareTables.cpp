#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace PSQT
{
    const FeatureType* psqt_data;


    INCBIN(FeatureType, EmbeddedPSQT, PSQT_Default_File);


    void init()
    {
        psqt_data = nullptr;
        load(PSQT_Default_File);
    }

    
    void load(std::string file)
    {
        if (psqt_data)
            clean();

        if (file == "" || file == PSQT_Default_File)
        {
            psqt_data = gEmbeddedPSQTData;
        }
        else
        {
            std::ifstream input(file, std::ios_base::binary);
            if (!input.is_open())
            {
                std::cerr << "Failed to open PSQ input file!" << std::endl;
                std::abort();
            }

            psqt_data = new FeatureType[NUM_FEATURES];
    
            if(!input.read(const_cast<char*>(reinterpret_cast<const char*>(psqt_data)), NUM_FEATURES * sizeof(FeatureType)))
            {
                std::cerr << "Failed to read PSQ data!" << std::endl;
                std::abort();
            }    
        }
    }


    void clean()
    {
        if (psqt_data != gEmbeddedPSQTData)
            delete[] psqt_data;
        psqt_data = nullptr;
    }
}


Score piece_square(PieceType piece, Square square, Turn turn, Square king_sq, Turn king_turn)
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

    return PSQT::psqt_data[index];
}