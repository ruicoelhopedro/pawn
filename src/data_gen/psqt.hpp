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
        static constexpr std::size_t NUM_MAX_FEATURES = 128;

        using Feature = uint32_t;
        using FeatureCount = uint8_t;
        using Eval = int16_t;
        using Result = int8_t;
        using PieceColor = int8_t;

        Feature features[NUM_MAX_FEATURES];
        PieceColor color[NUM_MAX_FEATURES];
        FeatureCount count;
        Eval eval;
        Result result;

        FeatureSample(const Board& board, Eval eval_input, Color result_input);

        void write(FileFormat& file);

    private:
        void push(Feature feature, PieceColor pc);
    };


    enum Phase
    {
        MG = 0,
        EG = 1
    };

    class Mapper
    {
    public:
        static constexpr std::size_t FEATURE_DIMS[] = { 2, NUM_SQUARES, NUM_PIECE_TYPES - 1, NUM_SQUARES / 2, NUM_COLORS };
        static constexpr std::size_t N_FEATURES = FEATURE_DIMS[0] * FEATURE_DIMS[1] * FEATURE_DIMS[2] * FEATURE_DIMS[3] * FEATURE_DIMS[4];

        static std::size_t map(Phase phase, Turn turn, PieceType piece, Square square, Square king_sq);
    };



    void gen_data_psqt(std::istringstream& stream);


    void games_to_psq_data(std::istringstream& stream);
}