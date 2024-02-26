#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include "MoveOrder.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "syzygy/syzygy.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>



Thread::Thread(int id, ThreadPool& pool)
    : m_id(id),
      m_pool(pool),
      m_status(ThreadStatus::STARTING),
      m_tb_hits(0),
      m_nodes_searched(0),
      m_multiPV(UCI::Options::MultiPV)
{
    m_thread = std::thread(&Thread::thread_loop, this);

    // Wait here until the thread has started
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cvar.wait(lock, [this]{ return m_status != ThreadStatus::STARTING; });
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
    int hashfull = m_pool.tt().hashfull();
    double elapsed = m_pool.m_time.elapsed();
    uint64_t nodes = m_pool.nodes_searched();
    uint64_t tb_hits = m_pool.tb_hits();

    // Output information
    for (int iPV = 0; iPV < UCI::Options::MultiPV; iPV++)
        m_multiPV[iPV].write_pv(m_position.board(), iPV, hashfull, nodes, tb_hits, elapsed);
}


int Thread::id() const { return m_id; }
Depth Thread::root_depth() const { return m_root_depth; }
bool Thread::is_main() const { return m_id == 0; }
ThreadPool& Thread::pool() const { return m_pool; }
const Search::SearchTime& Thread::time() const { return m_pool.m_time; }
const Search::Limits& Thread::limits() const { return m_pool.m_limits; }


void Thread::tb_hit() { m_tb_hits.fetch_add(1, std::memory_order_relaxed); }



ThreadPool::ThreadPool(int n_threads, int hash_size_mb)
    : m_tt(hash_size_mb),
      m_threads(0),
      m_status(ThreadStatus::WAITING)
{
    for (int i = 0; i < n_threads; i++)
        m_threads.push_back(std::make_unique<Thread>(i, *this));
}


ThreadPool::~ThreadPool()
{
    kill_threads();
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

    // Ensure position is correct in all threads
    update_position_threads();
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
    m_tt.new_search();
    m_status = ThreadStatus::SEARCHING;
    m_limits = limits;

    // Estimate search time
    update_time(timer, limits);

    // Tablebases probe
    Syzygy::Root = Syzygy::RootPos(m_position);

    // Clear debug data (if any)
    if constexpr (Debug::Enabled)
        Debug::clear_debug_data();

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


void ThreadPool::wait(Thread* skip)
{
    for (auto& thread : m_threads)
        if (thread.get() != skip)
            thread->wait();
}


void ThreadPool::clear()
{
    m_tt.clear();
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
        m_time.init(timer, std::max(1, limits.movetime - UCI::Options::MoveOverhead), limits.ponder);

    // With clock time
    else if (limits.time[turn] >= 0)
    {
        // Number of expected remaining moves
        int n_expected_moves = limits.movestogo >= 0 ? std::min(30, limits.movestogo) : 30;
        int time_remaining = limits.time[turn] + limits.incr[turn] * (n_expected_moves - 1);

        // This move will use 1/n_expected_moves of the remaining time
        int optimum = time_remaining / n_expected_moves - UCI::Options::MoveOverhead;
        int maximum = std::min(8 * limits.time[turn] / 10, optimum * 2) - UCI::Options::MoveOverhead;
        m_time.init(timer, std::max(1, maximum), std::max(1, optimum), limits.ponder);
    }

    // No time management
    else
        m_time.init(timer, limits.ponder);
}


const std::atomic<ThreadStatus>& ThreadPool::status() const { return m_status; }


Position& ThreadPool::position() { return m_position; }
Position ThreadPool::position() const { return m_position; }


int64_t ThreadPool::tb_hits() const
{
    int64_t total = 0;
    for (auto& thread : m_threads)
        total += thread->m_tb_hits.load(std::memory_order_relaxed);
    return total;
}


int64_t ThreadPool::nodes_searched() const
{
    int64_t total = 0;
    for (auto& thread : m_threads)
        total += thread->m_nodes_searched.load(std::memory_order_relaxed);
    return total;
}


int ThreadPool::size() const { return m_threads.size(); }


Thread* ThreadPool::get_best_thread() const
{
    // Get minimum score
    Score min_score = SCORE_INFINITE;
    for (auto& thread : m_threads)
        min_score = std::min(min_score, thread->m_multiPV[0].score());

    // Voting function
    auto thread_votes = [min_score](const Thread* thread)
    {
        return (thread->m_multiPV[0].score() - min_score + 20) * thread->m_multiPV[0].depth;
    };

    // Build votes for each thread
    Thread* best_thread = m_threads.front().get();
    int most_voted = thread_votes(best_thread);
    for (auto& thread : m_threads)
    {
        int vote = thread_votes(thread.get());
        if (vote > most_voted)
        {
            best_thread = thread.get();
            most_voted = vote;
        }
    }

    return best_thread;
}


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
    else if (Syzygy::Root.in_tb())
    {
        // Only select the top TB-scored moves (and make the selection aware of MultiPV)
        int num_tb_moves = std::max(Syzygy::Root.num_preserving_moves(), UCI::Options::MultiPV);
        m_root_moves.clear();
        for (int idx = 0; idx < num_tb_moves; idx++)
            m_root_moves.push(Syzygy::Root.ordered_moves(idx));
    }

    // Check for aborted search if game has ended
    if (m_root_moves.length() == 0 || m_position.is_draw(false))
    {
        if (main_thread)
        {
            // Even in this case, output a score and bestmove
            std::cout << "info depth 0 score " << (m_position.in_check() ? "mate 0" : "cp 0") << std::endl;
            std::cout << "bestmove (0000)" << std::endl;
            
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

    // Time management stuff
    int best_move_changes = 0;
    Move last_best_move = MOVE_NULL;
    int iter_since_best_move_change = 0;
    Score average_score = 0;

    // Clear data
    m_tb_hits.store(0);
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
            Depth depth = iDepth + m_id % 4;
            m_root_depth = depth;
            aspiration_search(m_position, pv, depth, data);

            // Timeout?
            if (timeout() && iDepth > 1)
                break;

            // Remove the bestmove we found from the root moves list
            m_root_moves.pop(std::find(m_root_moves.begin(), m_root_moves.end(), *pv.pv));

            // Sort Pv lines by depth and score
            std::stable_sort(m_multiPV.begin(), m_multiPV.end(),
                             [](Search::MultiPVData a, Search::MultiPVData b)
                             {
                                 return a.depth >= b.depth && a.score() > b.score();
                             });

            // Output all searched Pv lines
            if (main_thread && (iPv + 1 == maxPv || time.elapsed() > 3))
                output_pvs();
        }

        // Timeout?
        if (timeout())
            break;

        // Keep track of best move changes
        if (*m_multiPV.front().pv != last_best_move)
        {
            last_best_move = *m_multiPV.front().pv;
            iter_since_best_move_change = 0;
            best_move_changes++;
        }
        else
            iter_since_best_move_change++;

        // Update best score and its average
        Score best_score = m_multiPV.front().score();
        average_score = (best_score + 9 * average_score) / 10;

        // Additional task for main thread: check if we need to stop
        if (main_thread)
        {
            // Additional time stopping conditions
            Score score = m_multiPV.front().score();
            if (time.time_management() &&
                !limits.ponder &&
                !limits.infinite)
            {
                double optimum = time.optimum();

                // Increase time if the score is falling
                optimum *= std::clamp(1.0 + (average_score - best_score) / 100.0, 1.0, 1.75);

                // Adjust time based on the stability of the best move
                double stability = std::clamp(1.0 - iter_since_best_move_change / (2.0 * iDepth), 0.75, 1.0);
                double instability = std::clamp(0.9 + best_move_changes / (2.0 * iDepth), 1.0, 1.5);
                optimum *= stability * instability;

                // Compute corrected remaining time
                double remaining = optimum - time.elapsed();

                // Do we expect not to have time for one more iteration?
                if (remaining < timer_depth.elapsed())
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

        // Wait for other threads to finish
        m_pool.wait(this);

        // Select new best thread
        Thread* best_thread = m_pool.get_best_thread();

        // If best thread is not the main thread, re-send the last PV line
        if (best_thread != this)
            best_thread->output_pvs();

        // Fetch best and ponder moves from best Pv line
        Move* best_pv = best_thread->m_multiPV.front().pv;
        Move bestmove = *best_pv;
        Move pondermove = *(best_pv + 1);

        // If we get no pondermove from the PV, use the TT to try to guess a move to ponder
        if (pondermove == MOVE_NULL)
        {
            m_position.make_move(bestmove);
            TranspositionEntry* entry = nullptr;
            if (m_pool.tt().query(m_position.hash(), &entry) && m_position.board().legal(entry->hash_move()))
                pondermove = entry->hash_move();
            m_position.unmake_move();
        }

        // Mandatory output to the GUI
        std::cout << "bestmove " << m_position.board().to_uci(bestmove);
        if (pondermove != MOVE_NULL)
            std::cout << " ponder " << m_position.board().to_uci(pondermove);
        std::cout << std::endl;

        // Debug prints
        if constexpr (Debug::Enabled)
            Debug::print_debug_data();
    }
}
