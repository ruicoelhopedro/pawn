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

struct SearchResult
{
    Score score;
    Move bestmove;

    inline SearchResult()
        : score(0), bestmove(MOVE_NULL)
    {}

    inline SearchResult(Score s, Move m)
        : score(s), bestmove(m)
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

public:
    Piece get_piece_at(Square s) const;
    Turn get_turn() const;
    Square get_ep_square() const;
    uint8_t get_castle_rights() const;
    uint8_t get_half_move() const;

    BinaryBoard() = default;
    BinaryBoard(const Board& board);
    std::string fen() const;
};


struct BinaryNode
{
    Move move;
    int16_t score;

    BinaryNode() = default;

    inline BinaryNode(Move m, Score s)
        : move(m), score(s)
    {}
};


struct BinaryGame
{
    bool started;
    BinaryBoard starting_pos;
    std::vector<BinaryNode> nodes;

    inline BinaryGame()
        : started(false)
    {}

    static bool read(std::ifstream& stream, BinaryGame& result);

    static bool read(std::ifstream& stream, BinaryGame& result, std::size_t game_size);

    inline void begin(const Board& board)
    {
        started = true;
        starting_pos = BinaryBoard(board);
        nodes.clear();
    }

    inline void push(Move m, Score s) { nodes.push_back(BinaryNode(m, s)); }

    void write(std::ofstream& stream);

    void export_pgn(std::ofstream& stream) const;
};


std::vector<std::string> read_fens(std::string file_name);