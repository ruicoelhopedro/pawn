#pragma once
#include "incbin/incbin.h"
#include "Types.hpp"

#define NNUE_Default_File "nnue-897e7deab9e3.dat"


namespace NNUE
{

    using Weight = int16_t;
    using Feature = int;
    constexpr int SCALE_FACTOR = 1024;
    constexpr std::size_t NUM_FEATURES = 20480;
    constexpr std::size_t NUM_ACCUMULATORS = 128;
    constexpr std::size_t NUM_MAX_ACTIVE_FEATURES = 30;
    
    enum Phase
    {
        MG = 0,
        EG = 1,
        NUM_PHASES
    };

    struct Net
    {
        Weight m_psqt[NUM_FEATURES][NUM_PHASES];
        Weight m_sparse_layer[NUM_FEATURES][NUM_ACCUMULATORS];
        Weight m_bias[NUM_ACCUMULATORS];
        Weight m_dense[NUM_PHASES][NUM_ACCUMULATORS];
    };

    class Accumulator
    {
        int m_net[NUM_ACCUMULATORS];
        int m_psqt[NUM_PHASES];

    public:
        Accumulator();

        void clear();

        Feature get_feature(PieceType p, Square s, Square ks, Turn pt, Turn kt) const;

        void push(PieceType p, Square s, Square ks, Turn pt, Turn kt);

        void pop(PieceType p, Square s, Square ks, Turn pt, Turn kt);

        void push_features(std::size_t num_features, Feature* features);

        MixedScore eval() const;

        MixedScore eval_psq() const;

        bool operator==(const Accumulator& other) const;

        bool operator!=(const Accumulator& other) const;
    };

    extern const Net* nnue_net;

    void init();

    void load(std::string file);

    void clean();
}
