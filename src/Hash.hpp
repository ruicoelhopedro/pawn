#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include <vector>


using Age = uint8_t;


enum class EntryType
{
    EMPTY,
    UPPER_BOUND,
    LOWER_BOUND,
    EXACT
};


class TranspositionEntry
{
    static constexpr int GEN_DATA_BITS = 2;
    static constexpr uint8_t GEN_DATA_MASK = (1 << GEN_DATA_BITS) - 1;

    Hash m_hash;
    uint8_t m_depth;
    uint8_t m_type;
    int16_t m_score;
    Move m_best_move;
    int16_t m_static_eval;

    static Hash data_hash(Depth depth, int16_t score, uint8_t type, int16_t static_eval)
    {
        return (static_cast<Hash>(depth) << 0)
             | (static_cast<Hash>(score) << 8)
             | (static_cast<Hash>(type & GEN_DATA_MASK) << 40)
             | (static_cast<Hash>(static_eval) << 48);
    }

    Hash data_hash() const { return data_hash(depth(), score(), m_type, static_eval()); }

    uint8_t gen_type(Age age, EntryType type) { return (age << GEN_DATA_BITS) | static_cast<uint8_t>(type); }
    inline Age age() const { return static_cast<Age>(m_type >> GEN_DATA_BITS); }

public:
    TranspositionEntry()
        : m_type(gen_type(0, EntryType::EMPTY))
    {}

    inline bool query(Age age, Hash hash, TranspositionEntry** entry)
    {
        *entry = this;
        if (hash == this->hash())
        {
            // Bump up the age of this entry
            m_type = gen_type(age, type());
            return true;
        }
        return false;
    }
    inline void store(Hash hash, Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
    {
        m_hash = hash ^ data_hash(depth, score, gen_type(this->age(), type), static_eval);
        m_depth = depth;
        m_type = gen_type(this->age(), type);
        m_score = score;
        m_static_eval = static_eval;

        // Only replace the best move if we have a new one to store
        if (best_move != MOVE_NULL)
            m_best_move = best_move;
    }
    inline bool empty() const { return type() == EntryType::EMPTY; }

    inline Hash hash() const { return m_hash ^ data_hash(); }
    inline Depth depth() const { return m_depth; }
    inline EntryType type() const { return static_cast<EntryType>(m_type & 0b11); }
    inline Score score() const { return m_score; }
    inline Move hash_move() const { return m_best_move; }
    inline Score static_eval() const { return m_static_eval; }

    inline int value(int age) const { return m_depth - 255 * (age != this->age()); }
};


struct Bucket
{
    static constexpr int BUCKET_SIZE = 3;
    TranspositionEntry entries[BUCKET_SIZE];

    inline bool query(Age age, Hash hash, TranspositionEntry** entry_ptr)
    {
        // Search for this entry in the bucket
        for (std::size_t i = 1; i < BUCKET_SIZE; i++)
            if (entries[i].query(age, hash, entry_ptr))
                return true;

        // No entry found: find the least valuable entry and return it for replacement
        *entry_ptr = &entries[0];
        for (std::size_t i = 1; i < BUCKET_SIZE; i++)
            if (entries[i].depth() < (*entry_ptr)->depth())
                *entry_ptr = &entries[i];
        return false;
    }
};


class TranspositionTable
{
    std::vector<Bucket> m_table;
    std::size_t m_full;
    Age m_age;

    static std::size_t size_from_mb(std::size_t mb)   { return mb * 1024 / sizeof(Bucket) * 1024 + 1; }
    static std::size_t mb_from_size(std::size_t size) { return (size - 1) / 1024 * sizeof(Bucket) / 1024; }

    std::size_t index(Hash hash) const { return hash % m_table.size(); }

public:
    TranspositionTable()
        : TranspositionTable(0)
    {}

    TranspositionTable(std::size_t size, bool size_in_mb = true)
        : m_table(size_in_mb ? size_from_mb(size) : size),
          m_full(0),
          m_age(0)
    {}

    bool query(Hash hash, TranspositionEntry** entry_ptr)
    {
        return m_table[index(hash)].query(m_age, hash, entry_ptr);
    }


    void clear()
    {
        m_full = 0;
        m_age = 0;
        std::fill(m_table.begin(), m_table.end(), Bucket());
    }

    void new_search()
    {
        m_age++;
    }

    int max_size() const { return 262144; }

    void resize(std::size_t size_mb)
    {
        m_table = std::vector<Bucket>(size_from_mb(size_mb));
        m_full = 0;
    }

    int hashfull() const
    {
        int count = 0;
        for (int i = 0; i < 1000; i++)
            for (int j = 0; j < Bucket::BUCKET_SIZE; j++)
                if (!m_table[i].entries[j].empty())
                    count++;
        return count / Bucket::BUCKET_SIZE;
    }
};


extern TranspositionTable ttable;
