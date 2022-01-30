#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
#include "MoveOrder.hpp"
#include "Thread.hpp"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>


ThreadPool* pool;


Thread::Thread(int id, ThreadPool& pool)
    : m_id(id),
      m_pool(pool),
      m_status(ThreadStatus::WAITING),
      m_seldepth(0),
      m_nodes_searched(0),
      m_time(pool.m_time),
      m_limits(pool.m_limits),
      m_multiPV(Search::Parameters::multiPV)
{
    m_thread = std::thread(&Thread::thread_loop, this);
}


void Thread::thread_loop()
{
    while (m_status != ThreadStatus::QUITTING)
    {
        std::unique_lock<std::mutex> lock(m_pool.m_mutex);
        m_pool.m_cvar.wait(lock, [this]() { return m_pool.status() != m_status; });
        m_status = m_pool.status();
        lock.unlock();

        if (m_status == ThreadStatus::SEARCHING)
            search();
    }
}


void Thread::update_position(const Position& position) { m_position.update_from(position); }


void Thread::search()
{
    // Build search data and start iterative deepening
    Search::SearchData data(*this);
    Search::iter_deepening(m_position, data);
}


int Thread::id() const { return m_id; }
bool Thread::is_main() const { return m_id == 0; }
ThreadPool& Thread::pool() const { return m_pool; }
const Search::Time& Thread::time() const { return m_time; }
const Search::Limits& Thread::limits() const { return m_limits; }
std::vector<Search::MultiPVData>& Thread::multiPVs() { return m_multiPV; }



ThreadPool::ThreadPool()
    : m_threads(0),
      m_status(ThreadStatus::WAITING)
{
    m_threads.push_back(std::make_unique<Thread>(0, *this));
}


void ThreadPool::send_signal(ThreadStatus signal)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_status = signal;
    m_cvar.notify_all();
}


void ThreadPool::kill_threads()
{
    send_signal(ThreadStatus::QUITTING);
    
    for (auto& thread : m_threads)
        if (thread->m_thread.joinable())
            thread->m_thread.join();
    m_status = ThreadStatus::WAITING;
}


void ThreadPool::resize(int n_threads)
{
    kill_threads();
    m_threads.clear();
    for (int i = 0; i < n_threads; i++)
        m_threads.push_back(std::make_unique<Thread>(i, *this));
}


void ThreadPool::update_position_threads()
{
    // Update position in search threads
    for (auto& thread : m_threads)
        thread->update_position(m_position);
}


void ThreadPool::search(const Search::Limits& limits, bool wait)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = ThreadStatus::SEARCHING;
    m_limits = limits;
    m_time = Search::update_time(m_position, limits);
    m_cvar.notify_all();
    lock.unlock();

    if (wait)
    {
        std::unique_lock<std::mutex> wait_lock(m_mutex);
        m_cvar.wait(wait_lock, [this]() { return m_status != ThreadStatus::SEARCHING; });
    }
}


void ThreadPool::stop()
{
    //std::lock_guard<std::mutex> lock(m_mutex);
    m_status = ThreadStatus::WAITING;
    m_cvar.notify_all();
}


void ThreadPool::ponderhit()
{
    m_time.ponderhit();
}


void ThreadPool::update_multiPV(int n)
{
    for (auto& thread : m_threads)
        thread->m_multiPV.resize(n);
}


const std::atomic<ThreadStatus>& ThreadPool::status() const { return m_status; }

Position& ThreadPool::position() { return m_position; }
Position ThreadPool::position() const { return m_position; }


int64_t ThreadPool::nodes_searched() const
{
    int64_t total = 0;
    for (auto& thread : m_threads)
        total += thread->m_nodes_searched;
    return total;
}


int ThreadPool::size() const { return m_threads.size(); }