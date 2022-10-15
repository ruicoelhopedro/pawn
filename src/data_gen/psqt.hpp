#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include "PieceSquareTables.hpp"
#include "data_gen/data_gen.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


namespace PSQT_DataGen
{
    struct FileFormat
    {
        std::ofstream rows;
        std::ofstream evals;
        std::ofstream colors;
        std::ofstream indices;
        std::ofstream results;

        FileFormat(int offset, std::string prefix);
    };



    struct FeatureSample
    {
        using Feature = uint16_t;
        using FeatureCount = uint8_t;
        using Eval = int16_t;
        using Result = int8_t;
        using PieceColor = int8_t;

        Feature features[NUM_SQUARES];
        PieceColor color[NUM_SQUARES];
        FeatureCount count;
        Eval eval;
        Result result;

        FeatureSample(const Board& board, Eval eval_input, Color result_input);

        void write(FileFormat& file);

    private:
        void push(Feature feature, Color pc);
    };



    class Mapper
    {
    public:
        static constexpr std::size_t FEATURE_DIMS[] = { NUM_SQUARES, NUM_PIECE_TYPES - 1, NUM_SQUARES / 2, NUM_COLORS };
        static constexpr std::size_t N_FEATURES = FEATURE_DIMS[0] * FEATURE_DIMS[1] * FEATURE_DIMS[2] * FEATURE_DIMS[3];

        static std::size_t map(Turn turn, PieceType piece, Square square, Square king_sq);

        static Turn turn(std::size_t index);
        static PieceType piece(std::size_t index);
        static Square square(std::size_t index);
        static Square king_sq(std::size_t index);
    };



    void gen_data_psqt(std::istringstream& stream);
}