#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
#include "MoveOrder.hpp"
#include "Search.hpp"
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
    QUITTING
};


class ThreadPool;


class Thread
{
    int m_id;
    ThreadPool& m_pool;
    Position m_position;
    std::thread m_thread;
    ThreadStatus m_status;

    void thread_loop();

    void search();

protected:
    friend class Search::SearchData;
    friend class ThreadPool;
    Depth m_seldepth;
    PvContainer m_pv;
    Histories m_histories;
    int64_t m_nodes_searched;
    Search::Time& m_time;
    Search::Limits& m_limits;
    std::vector<Search::MultiPVData> m_multiPV;


public:
    Thread(int id, ThreadPool& pool);

    void update_position(const Position& position);

    int id() const;
    bool is_main() const;
    ThreadPool& pool() const;
    const Search::Time& time() const;
    const Search::Limits& limits() const;
    std::vector<Search::MultiPVData>& multiPVs();
};


class ThreadPool
{
    Position m_position;
    std::vector<std::unique_ptr<Thread>> m_threads;

    void send_signal(ThreadStatus signal);


protected:
    friend class Thread;
    std::mutex m_mutex;
    Search::Time m_time;
    std::atomic<ThreadStatus> m_status;
    Search::Limits m_limits;
    std::condition_variable m_cvar;

public:
    ThreadPool();

    void resize(int n_threads);

    void update_position_threads();

    void search(Search::Limits limits, Search::Time time, bool wait = false);

    void stop();

    void kill_threads();

    void ponderhit();

    void update_multiPV(int n);

    const std::atomic<ThreadStatus>& status() const;

    Position& position();
    Position position() const;
    
    int64_t nodes_searched() const;
};


extern ThreadPool* pool;