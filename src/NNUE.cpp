#include "incbin/incbin.h"
#include "Types.hpp"
#include "NNUE.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <memory>
#include <immintrin.h>

namespace NNUE
{
    std::unique_ptr<Net> nnue_net;


    INCBIN(char, EmbeddedNNUE, NNUE_Default_File);


    void init()
    {
        nnue_net = std::make_unique<Net>();
        nnue_net->load(gEmbeddedNNUEData, gEmbeddedNNUESize);
    }


    void load(std::string file)
    {
        if (file == "" || file == NNUE_Default_File)
        {
            // Load the default embedded network
            nnue_net->load(gEmbeddedNNUEData, gEmbeddedNNUESize);
        }
        else
        {
            // Open the network file and check its size
            std::ifstream input(file, std::ios_base::binary);
            if (!input.is_open())
            {
                std::cerr << "Failed to open NNUE file!" << std::endl;
                std::abort();
            }
            if (input.seekg(0, std::ios_base::end).tellg() != gEmbeddedNNUESize)
            {
                std::cerr << "Failed to read NNUE file: wrong size!" << std::endl;
                std::abort();
            }

            // Read the file into a buffer
            input.seekg(0, std::ios_base::beg);
            std::vector<char> buffer(gEmbeddedNNUESize);
            if(!input.read(buffer.data(), gEmbeddedNNUESize))
            {
                std::cerr << "Failed to read NNUE file!" << std::endl;
                std::abort();
            }

            // Load the network
            nnue_net->load(buffer.data(), gEmbeddedNNUESize);
        }
    }


    void Net::load(const char* data, std::size_t buffer_size)
    {
        std::size_t pos = 0;

        // Helper function to read data from the buffer
        auto read = [&data, &pos, buffer_size](auto* x, std::size_t x_size)
        {
            if (pos + x_size > buffer_size)
            {
                std::cerr << "Failed to read NNUE data: not enough data" << std::endl;
                std::abort();
            }
            std::memcpy(x, data + pos, x_size);
            pos += x_size;
        };

        // Read the network from the buffer
        read(m_sparse_layer, sizeof(m_sparse_layer));
        read(m_psqt, sizeof(m_psqt));
        read(m_bias, sizeof(m_bias));
        read(m_dense, sizeof(m_dense));
        read(m_dense_bias, sizeof(m_dense_bias));

        // Ensure that we read the entire buffer
        if (pos != buffer_size)
        {
            std::cerr << "Failed to read NNUE data: too much data" << std::endl;
            std::abort();
        }
    }



    Feature Accumulator::get_feature(PieceType p, Square s, Square ks, Turn pt, Turn kt) const
    {
        // Vertical mirror for black kings
        if (kt == BLACK)
        {
            pt = ~pt;
            s = vertical_mirror(s);
            ks = vertical_mirror(ks);
        }

        // Horiziontal mirror if king on the files E to H
        if (file(ks) >= 4)
        {
            s = horizontal_mirror(s);
            ks = horizontal_mirror(ks);
        }

        // Compute corrected king index (since we are mirrored there are only 4 files)
        int ki = 4 * rank(ks) + file(ks);

        // Compute index
        return s
             + p  *  NUM_SQUARES
             + ki * (NUM_SQUARES * (NUM_PIECE_TYPES - 1))
             + pt * (NUM_SQUARES * (NUM_PIECE_TYPES - 1) * NUM_SQUARES / 2);
    }


    Accumulator::Accumulator() { clear(); }


    void Accumulator::clear()
    {
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] = nnue_net->m_bias[i];
        for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
            m_psqt[bucket] = 0;
    }


    void Accumulator::push(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        Feature idx = get_feature(p, s, ks, pt, kt);
#if defined(__AVX2__)
        constexpr std::size_t ELEM_CYCLE = 16;
        constexpr std::size_t NUM_CYCLES = NUM_ACCUMULATORS / ELEM_CYCLE;
        for (std::size_t cycle = 0; cycle < NUM_CYCLES; cycle++)
        {
            std::size_t pos = cycle * ELEM_CYCLE;
            auto accumulator = reinterpret_cast<__m256i*>(&m_net[pos]);
            auto weights = reinterpret_cast<const __m256i*>(&nnue_net->m_sparse_layer[idx][pos]);
            __m256i result = _mm256_add_epi16(_mm256_load_si256(accumulator),
                                              _mm256_load_si256(weights));
            _mm256_store_si256(accumulator, result);
        }
#else
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] += nnue_net->m_sparse_layer[idx][i];
#endif
        for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
            m_psqt[bucket] += nnue_net->m_psqt[idx][bucket];
    }


    void Accumulator::pop(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        Feature idx = get_feature(p, s, ks, pt, kt);
#if defined(__AVX2__)
        constexpr std::size_t ELEM_CYCLE = 16;
        constexpr std::size_t NUM_CYCLES = NUM_ACCUMULATORS / ELEM_CYCLE;
        for (std::size_t cycle = 0; cycle < NUM_CYCLES; cycle++)
        {
            std::size_t pos = cycle * ELEM_CYCLE;
            auto accumulator = reinterpret_cast<__m256i*>(&m_net[pos]);
            auto weights = reinterpret_cast<const __m256i*>(&nnue_net->m_sparse_layer[idx][pos]);
            __m256i result = _mm256_sub_epi16(_mm256_load_si256(accumulator),
                                              _mm256_load_si256(weights));
            _mm256_store_si256(accumulator, result);
        }
#else
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] -= nnue_net->m_sparse_layer[idx][i];
#endif
        for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
            m_psqt[bucket] -= nnue_net->m_psqt[idx][bucket];
    }


    void Accumulator::push_features(std::size_t num_features, Feature* features)
    {
        for (std::size_t i = 0; i < num_features; i++)
        {
#if defined(__AVX2__)
            constexpr std::size_t ELEM_CYCLE = 16;
            constexpr std::size_t NUM_CYCLES = NUM_ACCUMULATORS / ELEM_CYCLE;
            for (std::size_t cycle = 0; cycle < NUM_CYCLES; cycle++)
            {
                std::size_t pos = cycle * ELEM_CYCLE;
                auto accumulator = reinterpret_cast<__m256i*>(&m_net[pos]);
                auto weights = reinterpret_cast<const __m256i*>(&nnue_net->m_sparse_layer[features[i]][pos]);
                __m256i result = _mm256_add_epi16(_mm256_load_si256(accumulator),
                                                  _mm256_load_si256(weights));
                _mm256_store_si256(accumulator, result);
            }
#else
            for (std::size_t j = 0; j < NUM_ACCUMULATORS; j++)
                m_net[j] += nnue_net->m_sparse_layer[features[i]][j];
#endif
            for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
                m_psqt[bucket] += nnue_net->m_psqt[features[i]][bucket];
        }
    }


    Score Accumulator::eval(const Accumulator& stm, const Accumulator& ntm, int bucket)
    {
#if defined(__AVX2__)
        constexpr std::size_t ELEM_CYCLE = 16;
        constexpr std::size_t NUM_CYCLES = NUM_ACCUMULATORS / ELEM_CYCLE;
        const __m256i upper_clip = _mm256_set1_epi16(SCALE_FACTOR);
        const __m256i lower_clip = _mm256_setzero_si256();
        __m256i result = _mm256_setzero_si256();
        for (std::size_t cycle = 0; cycle < NUM_CYCLES; cycle++)
        {
            const std::size_t idx = cycle * ELEM_CYCLE;
            auto acc_stm = reinterpret_cast<const __m256i*>(&stm.m_net[idx]);
            auto acc_ntm = reinterpret_cast<const __m256i*>(&ntm.m_net[idx]);

            // Clipped ReLU on the accumulators
            __m256i acc_stm_reg = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(acc_stm), lower_clip), upper_clip);
            __m256i acc_ntm_reg = _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256(acc_ntm), lower_clip), upper_clip);

            // Affine transformation
            const std::size_t idx2 = idx + NUM_ACCUMULATORS;
            auto weights_stm = reinterpret_cast<const __m256i*>(&nnue_net->m_dense[bucket][idx]);
            auto weights_ntm = reinterpret_cast<const __m256i*>(&nnue_net->m_dense[bucket][idx2]);
            __m256i madd_reg = _mm256_hadd_epi32(_mm256_madd_epi16(acc_stm_reg, _mm256_load_si256(weights_stm)),
                                                 _mm256_madd_epi16(acc_ntm_reg, _mm256_load_si256(weights_ntm)));
            result = _mm256_add_epi32(result, madd_reg);
        }

        __m128i sums = _mm_hadd_epi32(_mm256_extracti128_si256(result, 0),
                                      _mm256_extracti128_si256(result, 1));
        int output = nnue_net->m_dense_bias[bucket] * SCALE_FACTOR
                   + _mm_extract_epi32(sums, 0)
                   + _mm_extract_epi32(sums, 1)
                   + _mm_extract_epi32(sums, 2)
                   + _mm_extract_epi32(sums, 3);
#else
        int output = nnue_net->m_dense_bias[bucket] * SCALE_FACTOR;
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
        {
            // Clipped ReLU on the accumulators
            int stm_acc = std::clamp(stm.m_net[i], int16_t(0), SCALE_FACTOR);
            int ntm_acc = std::clamp(ntm.m_net[i], int16_t(0), SCALE_FACTOR);

            // Net output
            int j = i + NUM_ACCUMULATORS;
            output += nnue_net->m_dense[bucket][i] * stm_acc
                    + nnue_net->m_dense[bucket][j] * ntm_acc;
        }
#endif

        // Build and return final score
        return Score(stm.m_psqt[bucket] - ntm.m_psqt[bucket] + output / SCALE_FACTOR);
    }


    Score Accumulator::eval_psq(const Accumulator& stm, const Accumulator& ntm, int bucket)
    {
        return Score(stm.m_psqt[bucket] - ntm.m_psqt[bucket]);
    }
    

    bool Accumulator::operator==(const Accumulator& other) const
    {
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            if (m_net[i] != other.m_net[i])
                return false;
        for (std::size_t i = 0; i < NUM_BUCKETS; i++)
            if (m_psqt[i] != other.m_psqt[i])
                return false;
        return true;
    }

    bool Accumulator::operator!=(const Accumulator& other) const
    {
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            if (m_net[i] != other.m_net[i])
                return true;
        for (std::size_t i = 0; i < NUM_BUCKETS; i++)
            if (m_psqt[i] != other.m_psqt[i])
                return true;
        return false;
    }
}
