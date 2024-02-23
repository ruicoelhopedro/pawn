#pragma once

#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include "MoveOrder.hpp"
#include "Search.hpp"
#include "Evaluation.hpp"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>


enum class ThreadStatus
{
    WAITING,
    SEARCHING,
    STARTING,
    QUITTING
};


class ThreadPool;


class Thread
{
    int m_id;
    ThreadPool& m_pool;
    std::mutex m_mutex;
    Position m_position;
    std::thread m_thread;
    ThreadStatus m_status;
    MoveList m_root_moves;
    std::condition_variable m_cvar;

    void thread_loop();

    void search();

protected:
    friend class Search::SearchData;
    friend class ThreadPool;
    Depth m_seldepth;
    Depth m_root_depth;
    Search::PvContainer m_pv;
    Histories m_histories;
    std::atomic_uint64_t m_tb_hits;
    std::atomic_uint64_t m_nodes_searched;
    std::vector<Search::MultiPVData> m_multiPV;

public:
    Thread(int id, ThreadPool& pool);

    void update_position(const Position& position);

    bool is_root_move(Move move) const;

    void output_pvs();

    bool timeout() const;

    void wake(ThreadStatus status);

    void wait();

    void clear();

    Depth root_depth() const;

    int id() const;
    bool is_main() const;
    ThreadPool& pool() const;
    const Search::Limits& limits() const;
    const Search::SearchTime& time() const;

    void tb_hit();
};


class ThreadPool
{
    Position m_position;
    TranspositionTable m_tt;
    std::vector<std::unique_ptr<Thread>> m_threads;

    void send_signal(ThreadStatus signal);

protected:
    friend class Thread;
    Search::Limits m_limits;
    Search::SearchTime m_time;
    std::atomic<ThreadStatus> m_status;

public:
    ThreadPool(int n_threads, int hash_size_mb);

    virtual ~ThreadPool();

    void resize(int n_threads);

    void update_position_threads();

    void search(const Search::Timer& timer, const Search::Limits& limits);

    void stop();

    void kill_threads();

    void ponderhit();

    bool pondering() const;

    void update_time(const Search::Timer& timer, const Search::Limits& limits);

    const std::atomic<ThreadStatus>& status() const;

    Position& position();
    Position position() const;
    
    int64_t tb_hits() const;
    int64_t nodes_searched() const;

    int size() const;

    void wait();

    void wait(Thread* skip);

    void clear();

    Thread* get_best_thread() const;

    inline Thread& front() { return *(m_threads.front()); }
    inline TranspositionTable& tt() { return m_tt; }
};
