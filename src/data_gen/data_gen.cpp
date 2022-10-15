#include "Types.hpp"
#include "Position.hpp"
#include "data_gen/data_gen.hpp"
#include <fstream>


BinaryBoard::BinaryBoard(const Board& board)
{
    for (int i = 0; i < NUM_SQUARES; i += 2)
        pieces[i / 2] = (uint8_t(board.piece(i)) << 4) | uint8_t(board.piece(i + 1));

    uint8_t compressed_ep = board.ep_square() == SQUARE_NULL ? 0x8 : file(board.ep_square());
    uint8_t castling = (board.castle_rights(WHITE, KINGSIDE)  << 0)
                     + (board.castle_rights(WHITE, QUEENSIDE) << 1)
                     + (board.castle_rights(BLACK, KINGSIDE)  << 2)
                     + (board.castle_rights(BLACK, QUEENSIDE) << 3);
    
    other = (uint8_t(board.turn())            << 0)
          | (compressed_ep                    << 1)
          | (castling                         << 5)
          | (uint8_t(board.half_move_clock()) << 9);
}


Piece BinaryBoard::get_piece_at(Square s) const
{
    return Piece((s % 2 == 0) ? (pieces[s / 2] >> 4) : (pieces[s / 2] & 0b1111));
}


Turn BinaryBoard::get_turn() const
{
    return Turn(other & 0b1);
}


Square BinaryBoard::get_ep_square(Turn t) const
{
    return make_square(t == WHITE ? 2 : 5, (other >> 1) & 0b1111);
}


uint8_t BinaryBoard::get_castle_rights() const
{
    return (other >> 5) & 0b1111;
}


uint8_t BinaryBoard::get_half_move() const
{
    return (other >> 9) & 0b1111111;
}


std::string BinaryBoard::fen() const
{
    std::string result;

    // Position
    for (int rank = 7; rank >= 0; rank--)
    {
        int space = 0;
        for (int file = 0; file < 8; file++)
        {
            Piece pc = get_piece_at(make_square(rank, file));
            if (pc == Piece::NO_PIECE)
                space++;
            else
            {
                if (space)
                    result += space;
                result += fen_piece(pc);
                space = 0;
            }
        }
        if (space)
            result += space;
        result += (rank > 0 ? '/' : ' ');
    }

    // Side to move
    Turn turn = get_turn();
    result += (turn == WHITE ? "w " : "b ");

    // Castling rights
    uint8_t rights = get_castle_rights();
    if (rights)
    {
        if (rights & 0b1)    result += "K";
        if (rights & 0b10)   result += "Q";
        if (rights & 0b100)  result += "k";
        if (rights & 0b1000) result += "q";
        result += " ";
    }
    else
        result += "- ";

    // Ep square
    Square ep_square = get_ep_square(turn);
    result += (ep_square == SQUARE_NULL ? "-" : get_square(ep_square)) + ' ';

    // Half- and full-move clocks
    result += std::to_string(int(get_half_move())) + " 1";

    return result;
}


std::vector<std::string> read_fens(std::string file_name)
{
    std::string line;
    std::vector<std::string> fens;
    std::ifstream file(file_name);
    assert(file.is_open() && "Failed to open FEN file!");
    while(std::getline(file, line))
        fens.push_back(line);
    return fens;
}