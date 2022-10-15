#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include <string>
#include <vector>


struct GamePosition
{
    Board board;
    Score score;
    Move bestmove;

    inline GamePosition(Board b, Score s, Move m)
        : board(b), score(s), bestmove(m)
    {}
};


struct GameResult
{
    Color result;
    std::vector<GamePosition> game;
};


class BinaryBoard
{
    uint8_t pieces[32];
    uint16_t other;

    Piece get_piece_at(Square s) const;
    Turn get_turn() const;
    Square get_ep_square(Turn t) const;
    uint8_t get_castle_rights() const;
    uint8_t get_half_move() const;

public:
    BinaryBoard(const Board& board);
    std::string fen() const;
};


std::vector<std::string> read_fens(std::string file_name);