#include "Types.hpp"
#include "Position.hpp"
#include "data_gen/texel.hpp"
#include "PieceSquareTables.hpp"
#include "data_gen/data_gen.hpp"
#include "Thread.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>


namespace Texel
{
    void score_eval_error(std::istringstream& stream)
    {
        // Required parameter
        std::string input_file;
        assert((stream >> input_file) && "Failed to parse book file name!");

        // Open files
        std::ifstream ifile(input_file);
        assert(ifile.is_open() && "Failed to open input file!");

        Accumulator accumulator = 0;
        Accumulator num_positions = 0;

        // Read entire file
        BinaryGame game;
        while(BinaryGame::read(ifile, game))
        {
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                    {
                        // Score the loss if position verifies:
                        // i) Not in check
                        // ii) Best move not a capture
                        // iii) Best move not a promotion
                        // iv) No excessively large scores
                        if (!board.checkers() &&
                            !node.move.is_capture() &&
                            !node.move.is_promotion() &&
                            abs(node.score) < 2000)
                        {
                            int error = node.score - pool->front().evaluate<false>(board);
                            error = std::clamp(error, -1000, 1000);
                            accumulator += Accumulator(error * error);
                            num_positions++;
                        }

                        // Make the move
                        assert(board.legal(node.move) && "Illegal move!");
                        board = board.make_move(node.move);
                    }
        }

        // Compute final loss
        double loss = double(accumulator) / num_positions;
        std::cout << loss << std::endl;
    }


    void score_texel(std::istringstream& stream)
    {
        // Required parameters
        float K;
        std::string input_file;
        assert((stream >> K) && "Failed to parse K constant!");
        assert((stream >> input_file) && "Failed to parse book file name!");

        // Open files
        std::ifstream ifile(input_file);
        assert(ifile.is_open() && "Failed to open input file!");

        double accumulator = 0;
        Accumulator num_positions = 0;

        auto sigmoid = [K](int score) { return 1.0 / (1 + std::pow(10, -score * K / 400)); };

        // Read entire file
        BinaryGame game;
        while(BinaryGame::read(ifile, game))
        {
            // Find winner
            Color result = Color(game.nodes.back().score);

            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                    {
                        // Score the loss if position verifies:
                        // i) Not in check
                        // ii) Best move not a capture
                        // iii) Best move not a promotion
                        if (!board.checkers() &&
                            !node.move.is_capture() &&
                            !node.move.is_promotion())
                        {
                            double error = ((result + 1.0) / 2 - sigmoid(pool->front().evaluate<false>(board)));
                            accumulator += error * error;
                            num_positions++;
                        }

                        // Make the move
                        assert(board.legal(node.move) && "Illegal move!");
                        board = board.make_move(node.move);
                    }
        }

        // Compute final loss
        double loss = accumulator / num_positions;
        std::cout << loss << std::endl;
    }
}
