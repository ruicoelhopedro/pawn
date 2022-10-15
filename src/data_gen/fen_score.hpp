#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include "data_gen/data_gen.hpp"
#include <fstream>
#include <string>


namespace FEN_Scores
{
    struct FENFileFormat
    {
        std::ofstream file;

        FENFileFormat() = default;
        FENFileFormat(std::string prefix);
        FENFileFormat(int depth, std::string prefix);

        void write(const Position& pos, const std::string& fen, Score s);
        void write(const Position& pos, const std::string& fen, Move m, Score s);
    };


    struct BinaryFileFormat
    {
        std::ofstream file;

        BinaryFileFormat() = default;
        BinaryFileFormat(std::string prefix);
        BinaryFileFormat(int depth, std::string prefix);

        void write(const Position& pos, const std::string& fen, Score s);
        void write(const Position& pos, const std::string& fen, Move m, Score s);
    };


    void score_fens(std::istringstream& stream);

    void evaluate_fens(std::istringstream& stream);
}