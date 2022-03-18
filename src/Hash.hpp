#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include <vector>


class TableEntry
{
    virtual bool query(Hash hash, TableEntry** entry) = 0;
    virtual void store(Hash hash, Depth depth) = 0;
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

    inline bool query(Hash hash, PerftEntry** entry)
    {
        *entry = this;
        return hash == m_hash;
    }
    inline void store(Hash hash, Depth depth, uint64_t n_nodes)
    {
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
    Hash m_hash;
    Depth m_depth;
    uint8_t m_type;
    Score m_score;
    Move m_best_move;
    Score m_static_eval;

    static Hash data_hash(Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
    {
        return (static_cast<Hash>(depth) << 0)
             | (static_cast<Hash>(score) << 8)
             | (static_cast<Hash>(best_move.to_int()) << 24)
             | (static_cast<Hash>(type) << 40)
             | (static_cast<Hash>(static_eval) << 48);
    }

    Hash data_hash() const { return data_hash(depth(), score(), hash_move(), type(), static_eval()); }

    uint8_t gen_type(EntryType type) { return static_cast<uint8_t>(type); }

public:
    TranspositionEntry()
        : m_type(gen_type(EntryType::EMPTY))
    {}

    inline bool query(Hash hash, TranspositionEntry** entry)
    {
        *entry = this;
        return hash == (m_hash ^ data_hash());
    }
    inline void store(Hash hash, Depth depth, Score score, Move best_move, EntryType type, Score static_eval)
    {
        Hash old_hash = m_hash ^ data_hash();
        if (depth >= m_depth || hash != old_hash)
        {
            m_hash = hash ^ data_hash(depth, score, best_move, type, static_eval);
            m_depth = depth;
            m_type = gen_type(type);
            m_score = score;
            m_best_move = best_move;
            m_static_eval = static_eval;
        }
    }
    bool empty() const { return type() == EntryType::EMPTY; }

    inline Hash hash() const { return m_hash; }
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
    std::size_t m_full;

    static std::size_t size_from_mb(std::size_t mb)   { return mb * 1024 / sizeof(Entry) * 1024 + 1; }
    static std::size_t mb_from_size(std::size_t size) { return (size - 1) / 1024 * sizeof(Entry) / 1024; }

    std::size_t index(Hash hash) const { return hash % m_table.size(); }

public:
    HashTable()
        : HashTable(0)
    {}

    HashTable(std::size_t size_mb)
        : m_table(size_from_mb(size_mb)),
          m_full(0)
    {}

    template<typename EntryReturn>
    bool query(Hash hash, EntryReturn** entry_ptr)
    {
        return m_table[index(hash)].query(hash, entry_ptr);
    }

    template<typename... Args>
    void store(Hash hash, Args... args)
    {
        auto& entry = m_table[index(hash)];
        m_full += entry.empty();
        entry.store(hash, args...);
    }

    void clear()
    {
        m_full = 0;
        std::fill(m_table.begin(), m_table.end(), Entry());
    }

    int max_size() const { return 262144; }

    void resize(std::size_t size_mb)
    {
        m_table = std::vector<Entry>(size_from_mb(size_mb));
        m_full = 0;
    }

    int hashfull() const
    {
        return m_full * 1000 / m_table.size();
    }
};


extern HashTable<TranspositionEntry> ttable;
extern HashTable<PerftEntry> perft_table;