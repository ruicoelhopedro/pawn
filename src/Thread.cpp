#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include "MoveOrder.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include <algorithm>
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
      m_nodes_searched(0),
      m_multiPV(UCI::Options::MultiPV)
{
    m_thread = std::thread(&Thread::thread_loop, this);
}


void Thread::thread_loop()
{
    while (m_status != ThreadStatus::QUITTING)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_status = ThreadStatus::WAITING;
        m_cvar.notify_one();
        m_cvar.wait(lock, [this]() { return m_status != ThreadStatus::WAITING; });
        lock.unlock();

        if (m_status == ThreadStatus::SEARCHING)
            search();
    }
}


void Thread::wait()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cvar.wait(lock, [this]{ return m_status != ThreadStatus::SEARCHING; });
}


void Thread::clear()
{
    m_histories.clear();
}


void Thread::wake(ThreadStatus status)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_status = status;
    m_cvar.notify_one();
}


void Thread::update_position(const Position& position) { m_position = position; }


bool Thread::is_root_move(Move move) const
{
    return std::find(m_root_moves.begin(), m_root_moves.end(), move) != m_root_moves.end();
}


void Thread::output_pvs()
{
    double elapsed = m_pool.m_time.elapsed();
    uint64_t nodes = m_pool.nodes_searched();

    // Output information
    for (int iPV = 0; iPV < UCI::Options::MultiPV; iPV++)
        m_multiPV[iPV].write_pv(iPV, nodes, elapsed);
}


int Thread::id() const { return m_id; }
bool Thread::is_main() const { return m_id == 0; }
ThreadPool& Thread::pool() const { return m_pool; }
const Search::SearchTime& Thread::time() const { return m_pool.m_time; }
const Search::Limits& Thread::limits() const { return m_pool.m_limits; }



ThreadPool::ThreadPool()
    : m_threads(0),
      m_status(ThreadStatus::WAITING)
{
    m_threads.push_back(std::make_unique<Thread>(0, *this));
}


void ThreadPool::send_signal(ThreadStatus signal)
{
    for (auto& thread : m_threads)
        thread->wake(signal);
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


void ThreadPool::search(const Search::Timer& timer, const Search::Limits& limits)
{
    // Ensure all threads are stopped before we start searching
    this->wait();

    // Set the search data before waking the threads
    ttable.new_search();
    m_status = ThreadStatus::SEARCHING;
    m_limits = limits;

    // Estimate search time
    update_time(timer, limits);

    // Wake threads
    send_signal(ThreadStatus::SEARCHING);
}


void ThreadPool::stop()
{
    m_status = ThreadStatus::WAITING;
}


void ThreadPool::wait()
{
    for (auto& thread : m_threads)
        thread->wait();
}


void ThreadPool::clear()
{
    for (auto& thread : m_threads)
        thread->clear();
}


void ThreadPool::ponderhit()
{
    m_time.ponderhit();
}


bool ThreadPool::pondering() const
{
    return m_time.pondering();
}


void ThreadPool::update_time(const Search::Timer& timer, const Search::Limits& limits)
{
    Turn turn = m_position.get_turn();

    // Fixed movetime
    if (limits.movetime >= 0)
        m_time.init(timer, limits.movetime, limits.ponder);

    // With clock time
    else if (limits.time[turn] >= 0)
    {
        // Number of expected remaining moves
        int n_expected_moves = limits.movestogo >= 0 ? std::min(30, limits.movestogo) : 30;
        int time_remaining = limits.time[turn] + limits.incr[turn] * (n_expected_moves - 1);

        // This move will use 1/n_expected_moves of the remaining time
        int movetime = time_remaining / n_expected_moves;
        m_time.init(timer, movetime, limits.ponder);
    }

    // No time management
    else
        m_time.init(timer, limits.ponder);
}


const std::atomic<ThreadStatus>& ThreadPool::status() const { return m_status; }


Position& ThreadPool::position() { return m_position; }
Position ThreadPool::position() const { return m_position; }


int64_t ThreadPool::nodes_searched() const
{
    int64_t total = 0;
    for (auto& thread : m_threads)
        total += thread->m_nodes_searched.load(std::memory_order_relaxed);
    return total;
}


int ThreadPool::size() const { return m_threads.size(); }


bool Thread::timeout() const
{
    // Aborted search
    if (m_pool.status().load(std::memory_order_relaxed) != ThreadStatus::SEARCHING)
        return true;

    // Never timeout in ponder mode
    if (m_pool.m_time.pondering())
        return false;

    // Remaining time
    if (time().remaining() <= 0)
        return true;

    // Approximate number of nodes
    if (m_nodes_searched.load(std::memory_order_relaxed) > m_pool.m_limits.nodes / m_pool.size())
        return true;

    return false;
}


void Thread::search()
{
    bool main_thread = is_main();
    const Search::Limits& limits = m_pool.m_limits;
    const Search::SearchTime& time = m_pool.m_time;

    // Generate root moves
    Move moves[NUM_MAX_MOVES];
    m_root_moves = MoveList(moves);
    m_position.board().generate_moves(m_root_moves, MoveGenType::LEGAL);

    // Filter root moves if searchmoves has been passed
    if (limits.searchmoves.size() > 0)
    {
        Move* m = m_root_moves.begin();
        while (m < m_root_moves.end())
            if (std::find(limits.searchmoves.begin(), limits.searchmoves.end(), *m) == limits.searchmoves.end())
                m_root_moves.pop(m);
            else
                m++;
    }

    // Check for aborted search if game has ended
    if (m_root_moves.length() == 0 || m_position.is_draw(false))
    {
        if (main_thread)
        {
            // Even in this case, output a score and bestmove
            std::cout << "info depth 0 score " << (m_position.in_check() ? "mate 0" : "cp 0") << std::endl;
            std::cout << "bestmove " << MOVE_NULL << std::endl;
            
            // Stop the search
            m_pool.stop();
        }
        return;
    }

    // Maximum PV lines
    int maxPv = std::min(m_root_moves.length(), UCI::Options::MultiPV);

    // Prepare multiPV data
    m_multiPV.resize(UCI::Options::MultiPV);
    std::fill(m_multiPV.begin(), m_multiPV.end(), Search::MultiPVData());

    // Clear data
    m_nodes_searched.store(0);

    // Iterative deepening
    for (int iDepth = 1;
            iDepth < NUM_MAX_DEPTH && (iDepth <= limits.depth || m_pool.pondering());
            iDepth++)
    {
        // Start depth timer
        Search::Timer timer_depth;

        // MultiPV loop
        for (int iPv = 0; iPv < maxPv; iPv++)
        {
            Search::SearchData data(*this);
            Search::MultiPVData& pv = m_multiPV[iPv];

            // Push the previous Pv move back to the root moves list, if any
            if (*pv.pv != MOVE_NULL)
                m_root_moves.push(*pv.pv);

            // Carry the aspirated search
            Depth depth = iDepth + m_id / 2;
            aspiration_search(m_position, pv, depth, data);

            // Timeout?
            if (timeout())
                break;

            // Remove the bestmove we found from the root moves list
            m_root_moves.pop(std::find(m_root_moves.begin(), m_root_moves.end(), *pv.pv));

            // Sort Pv lines by depth and score
            std::sort(m_multiPV.begin(), m_multiPV.end(),
                      [](Search::MultiPVData a, Search::MultiPVData b)
                      {
                          return a.depth >= b.depth && a.score > b.score;
                      });

            // Output all searched Pv lines
            if (main_thread)
                output_pvs();
        }

        // Timeout?
        if (timeout())
            break;

        // Additional task for main thread: check if we need to stop
        if (main_thread)
        {
            // Additional time stopping conditions
            Score score = m_multiPV.front().score;
            if (time.time_management() &&
                !limits.ponder &&
                !limits.infinite)
            {
                double remaining = time.remaining();
                // Do we expect not to have time for one more iteration?
                if (remaining > 0 && remaining < timer_depth.elapsed() * 1.5)
                    break;
            }

            // Stopping condition for mate search
            if (limits.mate >= 1 &&
                is_mate(score) &&
                mate_in(score) <= limits.mate)
                break;
        }
    }

    // Main thread is responsible for the bestmove output
    if (main_thread)
    {
        // While pondering or in infinite mode we should not send a premature bestmove, so
        // we park here until the search gets stopped or we get a ponderhit
        while (!timeout() && (m_pool.pondering() || limits.infinite)) {}

        // Stop the search
        m_pool.stop();

        // Fetch best and ponder moves from best Pv line
        Move* best_pv = m_multiPV.front().pv;
        Move bestmove = *best_pv;
        Move pondermove = *(best_pv + 1);

        // Mandatory output to the GUI
        std::cout << "bestmove " << bestmove;
        if (pondermove != MOVE_NULL)
            std::cout << " ponder " << pondermove;
        std::cout << std::endl;
    }
}

