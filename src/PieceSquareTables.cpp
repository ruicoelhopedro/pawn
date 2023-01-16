#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include <iostream>
#include <fstream>
#include <string>

namespace PSQT
{
    const Net* psqt_net;


    INCBIN(Net, EmbeddedPSQT, PSQT_Default_File);


    void init()
    {
        psqt_net = nullptr;
        load(PSQT_Default_File);
    }

    
    void load(std::string file)
    {
        if (psqt_net)
            clean();

        if (file == "" || file == PSQT_Default_File)
        {
            psqt_net = gEmbeddedPSQTData;
        }
        else
        {
            std::ifstream input(file, std::ios_base::binary);
            if (!input.is_open())
            {
                std::cerr << "Failed to open PSQ net input file!" << std::endl;
                std::abort();
            }

            psqt_net = new Net[NUM_FEATURES];
    
            if(!input.read(const_cast<char*>(reinterpret_cast<const char*>(psqt_net)), sizeof(Net)))
            {
                std::cerr << "Failed to read PSQ net!" << std::endl;
                std::abort();
            }    
        }
    }


    void clean()
    {
        if (psqt_net != gEmbeddedPSQTData)
            delete[] psqt_net;
        psqt_net = nullptr;
    }

    

    int Accumulator::index(PieceType p, Square s, Square ks, Turn pt, Turn kt)
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
            m_net[i] = psqt_net->m_bias[i];
        m_psqt[MG] = 0;
        m_psqt[EG] = 0;
    }


    void Accumulator::push(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        int idx = index(p, s, ks, pt, kt);
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] += psqt_net->m_sparse_layer[i][idx];
        m_psqt[MG] += psqt_net->m_psqt[MG][idx];
        m_psqt[EG] += psqt_net->m_psqt[EG][idx];
    }


    void Accumulator::pop(PieceType p, Square s, Square ks, Turn pt, Turn kt)
    {
        if (p == KING)
            return;

        int idx = index(p, s, ks, pt, kt);
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            m_net[i] -= psqt_net->m_sparse_layer[i][idx];
        m_psqt[MG] -= psqt_net->m_psqt[MG][idx];
        m_psqt[EG] -= psqt_net->m_psqt[EG][idx];
    }


    MixedScore Accumulator::eval() const
    {
        // Clipped ReLU on the accumulators
        int accumulator[NUM_ACCUMULATORS];
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
            accumulator[i] = std::clamp(m_net[i], 0, SCALE_FACTOR);

        // Net output
        int output[2] = { 0, 0 };
        for (std::size_t i = 0; i < NUM_ACCUMULATORS; i++)
        {
            output[MG] += psqt_net->m_dense[MG][i] * accumulator[i];
            output[EG] += psqt_net->m_dense[EG][i] * accumulator[i];
        }

        // Build and return final score
        int mg = (m_psqt[MG] + output[MG] / SCALE_FACTOR) ;
        int eg = (m_psqt[EG] + output[EG] / SCALE_FACTOR) ;
        return MixedScore(mg, eg);
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
