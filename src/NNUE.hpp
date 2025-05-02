#pragma once
#include "Types.hpp"


#define NNUE_Default_File "nnue-2de23dfb491f.dat"


namespace NNUE
{

    using Weight = int16_t;
    using Feature = uint16_t;
    constexpr int16_t SCALE_FACTOR = 1024;
    constexpr std::size_t NUM_FEATURES = 20480;
    constexpr std::size_t NUM_ACCUMULATORS = 256;
    constexpr std::size_t NUM_MAX_ACTIVE_FEATURES = 30;
    constexpr std::size_t NUM_BUCKETS = 4;

    struct Net
    {
        Weight m_sparse_layer[NUM_FEATURES][NUM_ACCUMULATORS];
        Weight m_psqt[NUM_FEATURES][NUM_BUCKETS];
        Weight m_bias[NUM_ACCUMULATORS];
        Weight m_dense[NUM_BUCKETS][2 * NUM_ACCUMULATORS];
        Weight m_dense_bias[NUM_BUCKETS];
    };

    class Accumulator
    {
        int16_t m_net[NUM_ACCUMULATORS];
        int16_t m_psqt[NUM_BUCKETS];

    public:
        Accumulator();

        void clear();

        static Feature get_feature(PieceType p, Square s, Square ks, Turn pt, Turn kt);

        void push(PieceType p, Square s, Square ks, Turn pt, Turn kt);

        void pop(PieceType p, Square s, Square ks, Turn pt, Turn kt);

        void push_features(std::size_t num_features, Feature* features);

        static Score eval(const Accumulator& stm, const Accumulator& ntm, int bucket);

        static Score eval_psq(const Accumulator& stm, const Accumulator& ntm, int bucket);

        bool operator==(const Accumulator& other) const;

        bool operator!=(const Accumulator& other) const;
    };

    void init();

    void load(std::string file);

    void clean();
}
