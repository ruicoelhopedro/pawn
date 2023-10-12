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
        m_psqt[MG] = 0;
        m_psqt[EG] = 0;
    }


    void Accumulator::push(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        Feature idx = get_feature(p, s, ks, pt, kt);
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] += nnue_net->m_sparse_layer[idx][i];
        m_psqt[MG] += nnue_net->m_psqt[idx][MG];
        m_psqt[EG] += nnue_net->m_psqt[idx][EG];
    }


    void Accumulator::pop(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        Feature idx = get_feature(p, s, ks, pt, kt);
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] -= nnue_net->m_sparse_layer[idx][i];
        m_psqt[MG] -= nnue_net->m_psqt[idx][MG];
        m_psqt[EG] -= nnue_net->m_psqt[idx][EG];
    }


    void Accumulator::push_features(std::size_t num_features, Feature* features)
    {
        for (std::size_t i = 0; i < num_features; i++)
        {
            for (std::size_t j = 0; j < NUM_ACCUMULATORS; j++)
                m_net[j] += nnue_net->m_sparse_layer[features[i]][j];
            m_psqt[MG] += nnue_net->m_psqt[features[i]][MG];
            m_psqt[EG] += nnue_net->m_psqt[features[i]][EG];
        }
    }


    MixedScore Accumulator::eval() const
    {
        int output[2] = { 0, 0 };
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
        {
            // Clipped ReLU on the accumulators
            int accumulator = std::clamp(m_net[i], 0, SCALE_FACTOR);

            // Net output
            output[MG] += nnue_net->m_dense[MG][i] * accumulator;
            output[EG] += nnue_net->m_dense[EG][i] * accumulator;
        }

        // Build and return final score
        int mg = (m_psqt[MG] + output[MG] / SCALE_FACTOR) ;
        int eg = (m_psqt[EG] + output[EG] / SCALE_FACTOR) ;
        return MixedScore(mg, eg);
    }


    MixedScore Accumulator::eval_psq() const
    {
        return MixedScore(m_psqt[MG], m_psqt[EG]);
    }
    

    bool Accumulator::operator==(const Accumulator& other) const
    {
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            if (m_net[i] != other.m_net[i])
                return false;
        if (m_psqt[MG] != other.m_psqt[MG] || m_psqt[EG] != other.m_psqt[EG])
            return false;
        return true;
    }

    bool Accumulator::operator!=(const Accumulator& other) const
    {
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            if (m_net[i] != other.m_net[i])
                return true;
        if (m_psqt[MG] != other.m_psqt[MG] || m_psqt[EG] != other.m_psqt[EG])
            return true;
        return false;
    }
}
