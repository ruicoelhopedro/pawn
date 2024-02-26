#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include <vector>


using Age = uint8_t;

class TableEntry
{
    virtual bool query(Age age, Hash hash, TableEntry** entry) = 0;
    virtual void store(Age age, Hash hash, Depth depth) = 0;
    virtual bool empty() const = 0;
};


class PerftEntry
{
    Hash m_hash;
    Depth m_depth;
    uint64_t m_nodes;

public:
    inline PerftEntry()
        : m_hash(0), m_depth(0), m_nodes(0)
    {}

    inline bool query(Age age, Hash hash, PerftEntry** entry)
    {
        (void)age;
        *entry = this;
        return hash == m_hash;
    }
    inline void store(Age age, Hash hash, Depth depth, uint64_t n_nodes)
    {
        (void)age;
        m_hash = hash;
        m_depth = depth;
        m_nodes = n_nodes;
    }
    inline bool empty() const
    {
        return m_depth == 0 && m_nodes == 0;
    }

    Hash hash() const { return m_hash; }
    Depth depth() const { return m_depth; }
    uint64_t n_nodes() const { return m_nodes; }
};


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
        : m_hash(0), m_depth(0), m_type(gen_type(0, EntryType::EMPTY)),
          m_score(SCORE_NONE), m_best_move(MOVE_NULL), m_static_eval(SCORE_NONE)
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
    inline void store(Age age, Hash hash, Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
    {
        bool replace = type == EntryType::EXACT
                    || age != this->age()
                    || depth > m_depth - 4;
        if (replace)
        {
            m_hash = hash ^ data_hash(depth, score, gen_type(age, type), static_eval);
            m_depth = depth;
            m_type = gen_type(age, type);
            m_score = score;
            m_static_eval = static_eval;

            // Only replace the best move if we have a new one to store
            if (best_move != MOVE_NULL)
                m_best_move = best_move;
        }
    }
    bool empty() const { return type() == EntryType::EMPTY; }

    inline Hash hash() const { return m_hash ^ data_hash(); }
    inline Depth depth() const { return m_depth; }
    inline EntryType type() const { return static_cast<EntryType>(m_type & 0b11); }
    inline Score score() const { return m_score; }
    inline Move hash_move() const { return m_best_move; }
    inline Score static_eval() const { return m_static_eval; }
};


template<typename Entry>
class HashTable
{
    std::vector<Entry> m_table;
    Age m_age;

    static std::size_t size_from_mb(std::size_t mb)   { return mb * 1024 / sizeof(Entry) * 1024 + 1; }
    static std::size_t mb_from_size(std::size_t size) { return (size - 1) / 1024 * sizeof(Entry) / 1024; }

    std::size_t index(Hash hash) const { return hash % m_table.size(); }

public:
    HashTable()
        : HashTable(0)
    {}

    HashTable(std::size_t size, bool size_in_mb = true)
        : m_table(size_in_mb ? size_from_mb(size) : size),
          m_age(0)
    {}

    template<typename EntryReturn>
    bool query(Hash hash, EntryReturn** entry_ptr)
    {
        return m_table[index(hash)].query(m_age, hash, entry_ptr);
    }

    template<typename... Args>
    void store(Hash hash, Args... args)
    {
        m_table[index(hash)].store(m_age, hash, args...);
    }

    void clear()
    {
        m_age = 0;
        std::fill(m_table.begin(), m_table.end(), Entry());
    }

    void new_search()
    {
        m_age++;
    }

    int max_size() const { return 262144; }

    void resize(std::size_t size_mb)
    {
        m_table = std::vector<Entry>(size_from_mb(size_mb));
    }

    int hashfull() const
    {
        int count = 0;
        for (int i = 0; i < 1000; i++)
            count += !m_table[index(i)].empty();
        return count;
    }

    void prefetch(Hash hash)
    {
        __builtin_prefetch(&m_table[index(hash)]);
    }
};

using TranspositionTable = HashTable<TranspositionEntry>;
using PerftTable = HashTable<PerftEntry>;