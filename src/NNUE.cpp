#include "incbin/incbin.h"
#include "Types.hpp"
#include "NNUE.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace NNUE
{
    const Net* nnue_net;


    INCBIN(Net, EmbeddedNNUE, NNUE_Default_File);


    void init()
    {
        nnue_net = nullptr;
        if (sizeof(Net) != gEmbeddedNNUESize)
        {
            std::cerr << "Invalid size of embedded network! Expected "
                      << sizeof(Net)
                      << ", found "
                      << gEmbeddedNNUESize
                      << std::endl;
            std::abort();
        }
        load(NNUE_Default_File);
    }

    
    void load(std::string file)
    {
        if (nnue_net)
            clean();

        if (file == "" || file == NNUE_Default_File)
        {
            nnue_net = gEmbeddedNNUEData;
        }
        else
        {
            std::ifstream input(file, std::ios_base::binary);
            if (!input.is_open())
            {
                std::cerr << "Failed to open NNUE file!" << std::endl;
                std::abort();
            }

            nnue_net = new Net;
    
            if(!input.read(const_cast<char*>(reinterpret_cast<const char*>(nnue_net)), sizeof(Net)))
            {
                std::cerr << "Failed to read NNUE file!" << std::endl;
                std::abort();
            }    
        }
    }


    void clean()
    {
        if (nnue_net != gEmbeddedNNUEData)
            delete nnue_net;
        nnue_net = nullptr;
    }

    

    Feature Accumulator::get_feature(PieceType p, Square s, Square ks, Turn pt, Turn kt)
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
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] += nnue_net->m_sparse_layer[idx][i];
        for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
            m_psqt[bucket] += nnue_net->m_psqt[idx][bucket];
    }


    void Accumulator::pop(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        Feature idx = get_feature(p, s, ks, pt, kt);
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] -= nnue_net->m_sparse_layer[idx][i];
        for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
            m_psqt[bucket] -= nnue_net->m_psqt[idx][bucket];
    }


    void Accumulator::push_features(std::size_t num_features, Feature* features)
    {
        for (std::size_t i = 0; i < num_features; i++)
        {
            for (std::size_t j = 0; j < NUM_ACCUMULATORS; j++)
                m_net[j] += nnue_net->m_sparse_layer[features[i]][j];
            for (std::size_t bucket = 0; bucket < NUM_BUCKETS; bucket++)
                m_psqt[bucket] += nnue_net->m_psqt[features[i]][bucket];
        }
    }


    Score Accumulator::eval(const Accumulator& stm, const Accumulator& ntm, int bucket)
    {
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
