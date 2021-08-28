#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include <vector>

enum class EntryType
{
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND
};

class TranspositionEntry
{
    Hash m_hash;
    Depth m_depth;
    EntryType m_type;
    Score m_score;
    Move m_best_move;
    Score m_static_eval;

    static Hash data_hash(Depth depth, Score score, Move best_move, EntryType type, Score static_eval);

public:
    TranspositionEntry();
    TranspositionEntry(Hash hash, Depth depth, Score score, Move best_move, EntryType type, Score static_eval);

    bool is_empty() const;
    EntryType type() const;
    Depth depth() const;
    Score score() const;
    Score static_eval() const;
    Hash hash() const;
    Move hash_move() const;
    void reset();
};


class PerftEntry
{
    Hash m_hash;
    Depth m_depth;
    int m_nodes;

public:
    PerftEntry();
    PerftEntry(Hash hash, Depth depth, int n_nodes);

    bool is_empty() const;
    Depth depth() const;
    Hash hash() const;
    int n_nodes() const;
    void reset();
};


template <class T>
class TranspositionTable
{
    std::vector<T> m_table;
    uint64_t m_size;
    uint64_t m_full;

public:
    TranspositionTable()
        : m_table(1), m_size(1), m_full(0)
    {}


    TranspositionTable(int64_t size_mb)
        : m_table(size_mb * 1024 / sizeof(T) * 1024 + 1), m_size(size_mb * 1024 / sizeof(T) * 1024 + 1), m_full(0)
    {}


    bool query(Hash hash, T** entry_ptr)
    {
        T* entry = &(m_table[hash % m_size]);
        *entry_ptr = entry;

        if (entry == nullptr || entry->is_empty() || entry->hash() != hash)
        {
            entry = nullptr;
            return false;
        }

        return true;
    }


    void store(T entry, bool force = false)
    {
        int index = entry.hash() % m_size;

        if (m_table[index].is_empty())
            m_full++;

        if (force || entry.hash() != m_table[index].hash() || entry.depth() >= m_table[index].depth())
            m_table[index] = entry;
    }


    void clear()
    {
        m_full = 0;
        for (auto& entry : m_table)
            entry = T();
    }


    int max_size() const
    {
        auto size = (m_table.max_size() - 1) / 1024 * sizeof(T) / 1024;
        return (size > INT32_MAX) ? INT32_MAX : size;
    }


    void resize(int size)
    {
        m_size = size * 1024 / sizeof(T) * 1024 + 1;
        m_table = std::vector<T>(m_size);
        m_full = 0;
    }


    int hashfull() const
    {
        return m_full * 1000 / m_size;
    }
};


extern TranspositionTable<TranspositionEntry> ttable;
extern TranspositionTable<PerftEntry> perft_table;
