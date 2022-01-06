#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
#include "MoveOrder.hpp"
#include "Search.hpp"
#include "Evaluation.hpp"
#include "Zobrist.hpp"
#include <atomic>
#include <chrono>
#include <vector>

namespace Search
{
    ThreadStatus status = ThreadStatus::WAITING;
    int64_t nodes_searched;
    Position* base_position;
    std::mutex mutex;
    std::condition_variable cvar;
    std::vector<std::unique_ptr<SearchThread>> threads;
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::vector<Move> multiPV_roots;
    std::vector<Score> multiPV_scores;
    std::vector<MoveList> multiPV_lists;


    namespace Parameters
    {
        int multiPV = 1;
        int n_threads = 1;
        Limits limits = Limits();
        bool ponder = false;
    }


    SearchData::SearchData(Histories& histories, int thread_id, int64_t& nodes_searched, Depth& seldepth, Move* pv, Move* prev_pv)
        : m_ply(0), m_nodes_searched(nodes_searched), m_seldepth(seldepth), m_flags(NONE),
          m_reductions(0), m_extensions(0), m_histories(histories), m_prev(nullptr),
          m_excluded_move(MOVE_NULL), m_thread_id(thread_id), m_static_eval(SCORE_NONE),
          m_move(MOVE_NULL), m_pv(pv), m_prev_pv(prev_pv), m_isPv(true)
    {
    }


    SearchData SearchData::next(Move move) const
    {
        SearchData result = *this;
        result.m_prev = this;
        result.m_move = move;
        // Next flags
        result.m_flags = NONE;
        if (m_flags & REDUCED)
            result.m_reductions++;
        if (m_flags & EXTENDED)
            result.m_extensions++;
        // Other parameters
        result.m_ply++;
        result.m_static_eval = SCORE_NONE;
        result.m_excluded_move = MOVE_NULL;
        // New pv location
        result.m_prev_pv++;
        result.m_pv += PV_LENGTH - m_ply;
        // Are we still in a PV line?
        result.m_isPv = m_isPv && move == *m_prev_pv;
        return result;
    }


    Histories& SearchData::histories() const
    {
        return m_histories;
    }


    int SearchData::reductions() const
    {
        return m_reductions;
    }


    int SearchData::extensions() const
    {
        return m_extensions;
    }


    int SearchData::ply() const
    {
        return m_ply;
    }


    Move SearchData::excluded_move() const
    {
        return m_excluded_move;
    }


    int SearchData::thread() const
    {
        return m_thread_id;
    }


    Score& SearchData::static_eval()
    {
        return m_static_eval;
    }


    Score SearchData::static_eval() const
    {
        return m_static_eval;
    }


    Move SearchData::last_move() const
    {
        return m_move;
    }


    const SearchData* SearchData::previous(int distance) const
    {
        const SearchData* curr = this;
        for (int i = 0; i < distance; i++)
            curr = curr->m_prev;
        return curr;
    }


    int64_t& SearchData::nodes_searched()
    {
        return m_nodes_searched;
    }


    Depth& SearchData::seldepth()
    {
        return m_seldepth;
    }


    bool SearchData::in_pv() const
    {
        return m_isPv;
    }


    Move* SearchData::pv()
    {
        return m_pv;
    }


    Move SearchData::pv_move()
    {
        return m_isPv ? *m_prev_pv : MOVE_NULL;
    }


    void SearchData::exclude(Move move)
    {
        m_excluded_move = move;
    }


    void SearchData::update_pv(Move best_move, Move* new_pv)
    {
        // Set the initial bestmove
        Move* dst = m_pv;
        *(dst++) = best_move;

        // Loop over remaining PV
        while (*new_pv != MOVE_NULL)
            *(dst++) = *(new_pv++);

        // Set last entry as null move for a stop condition
        *dst = MOVE_NULL;
    }


    void SearchData::accept_pv()
    {
        // Copy last PV from the table to the prev PV
        Move* src = m_pv;
        Move* dst = m_prev_pv;
        while (*src != MOVE_NULL)
            *(dst++) = *(src++);

        // Set last entry as null move as a stop condition
        *dst = MOVE_NULL;
    }


    void SearchData::clear_pv()
    {
        for (int i = 0; i < TOTAL_PV_LENGTH; i++)
            m_pv[i] = MOVE_NULL;
    }


    SearchFlags SearchData::flags() const
    {
        return m_flags;
    }


    SearchData& SearchData::operator|=(SearchFlags flag)
    {
        m_flags |= flag;
        return *this;
    }


    SearchData& SearchData::operator^=(SearchFlags flag)
    {
        m_flags ^= flag;
        return *this;
    }


    SearchData SearchData::operator|(SearchFlags flag) const
    {
        SearchData result = *this;
        result |= flag;
        return result;
    }


    SearchData SearchData::operator^(SearchFlags flag) const
    {
        SearchData result = *this;
        result ^= flag;
        return result;
    }



    SearchThread::SearchThread(int id)
        : m_id(id),
          m_histories(std::make_unique<Histories>()),
          m_pv(std::make_unique<PvContainer>()),
          m_nodes_searched(0),
          m_seldepth(0),
          m_data(*m_histories, id, m_nodes_searched, m_seldepth, m_pv->pv, m_pv->prev_pv),
          m_local_status(ThreadStatus::WAITING)
    {
        for (int i = 0; i < TOTAL_PV_LENGTH; i++)
            m_pv->pv[i] = MOVE_NULL;
        for (int i = 0; i < PV_LENGTH; i++)
            m_pv->prev_pv[i] = MOVE_NULL;
    }


    ThreadStatus SearchThread::receive_signal()
    {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this]() { return Search::status != m_local_status; });
        return Search::status;
    }


    void SearchThread::thread_loop()
    {
        m_local_status = ThreadStatus::WAITING;
        while (m_local_status != ThreadStatus::QUITTING)
        {
            // Wait for a signal
            m_local_status = ThreadStatus::WAITING;
            m_local_status = receive_signal();

            // Begin search
            if (m_local_status == ThreadStatus::SEARCHING)
                iter_deepening(m_position, m_data);
        }
    }


    void SearchThread::set_position(const Position& pos)
    {
        m_position.update_from(pos);
    }


    void SearchThread::start_search()
    {
        auto stored_status = m_local_status;
        m_local_status = ThreadStatus::SEARCHING;
        iter_deepening(m_position, m_data);
        m_local_status = stored_status;
    }


    Histories& SearchThread::histories()
    {
        return *m_histories;
    }


    SearchData& SearchThread::data()
    {
        return m_data;
    }


    ThreadStatus SearchThread::status() const
    {
        return m_local_status;
    }



    bool timeout()
    {
        if (status != ThreadStatus::SEARCHING)
            return true;

        // Never timeout in ponder mode
        if (Parameters::limits.ponder)
            return false;

        if (std::chrono::steady_clock::now() > end_time)
            return true;

        if (nodes_searched > Parameters::limits.nodes)
            return true;

        return false;
    }



    void update_time()
    {
        auto& limits = Parameters::limits;
        Turn turn = base_position->get_turn();

        // We use the initial movetime as reference
        int movetime = limits.movetime;

        // With clock time
        if (limits.time[turn] >= 0)
        {
            // Number of expected remaining moves
            int n_expected_moves = std::max(1, std::min(50, limits.movestogo));
            int time_remaining = limits.time[turn] + limits.incr[turn] * n_expected_moves;

            // This move will use 1/n_expected_moves of the remaining time
            movetime = std::min(movetime, time_remaining / n_expected_moves);
        }

        // Update end time accordingly
        start_time = std::chrono::steady_clock::now();
        if (!limits.ponder)
            end_time = start_time + std::chrono::milliseconds(movetime);
    }



    int time_elapsed()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start_time).count() / 1e6;
    }



    void signal_threads(ThreadStatus signal)
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            status = signal;
        }
        cvar.notify_all();

        // Wait until we are sure all threads received the signal
        int remaining = threads.size();
        while (remaining > 0)
        {
            cvar.notify_all();
            remaining = threads.size();
            for (auto& thread : threads)
                if (thread->status() == status)
                    remaining--;
        }
    }



    void set_num_threads(int n_threads)
    {
        Parameters::n_threads = n_threads;

        // Ensure all threads stop
        kill_search_threads();

        // Spawn new threads
        threads.resize(n_threads);
        status = ThreadStatus::WAITING;
        for (int i = 0; i < n_threads; i++)
        {
            threads[i] = std::make_unique<SearchThread>(i);
            auto& st = *threads[i];
            st.set_position(*base_position);
            st.thread = std::thread(&SearchThread::thread_loop, &st);
        }
    }



    void start_search_threads()
    {
        nodes_searched = 0;
        signal_threads(ThreadStatus::SEARCHING);
    }



    void stop_search_threads()
    {
        signal_threads(ThreadStatus::WAITING);
    }



    void kill_search_threads()
    {
        signal_threads(ThreadStatus::QUITTING);

        for (auto& thread : threads)
        {
            while (true)
            {
                cvar.notify_all();
                if (thread->thread.joinable())
                {
                    thread->thread.join();
                    break;
                }
            }
        }
    }



    void go_perft(Depth depth)
    {
        Position position = *base_position;
        int64_t n_nodes = perft<true>(position, depth);
        std::cout << "\nNodes searched: " << n_nodes << std::endl;
    }



    void get_pv(Position& position, SearchData& data, MoveList& pv)
    {
        pv.clear();
        Move* pos = data.pv();
        while (*pos != MOVE_NULL)
            pv.push(*(pos++));
    }



    Score aspiration_search(Position& position, Score init_score, Depth depth, Color color, SearchData& data)
    {
        constexpr Score starting_window = 25;
        Score l_window = starting_window;
        Score r_window = starting_window;

        // Initial search
        Score alpha = (depth <= 4) ? (-SCORE_INFINITE) : std::max(-SCORE_INFINITE, (init_score - l_window));
        Score beta  = (depth <= 4) ? ( SCORE_INFINITE) : std::min(+SCORE_INFINITE, (init_score + r_window));

        // Open windows for large values
        if (alpha < 1000)
            alpha = -SCORE_INFINITE;
        if (beta > 1000)
            beta = SCORE_INFINITE;

        // Special case for mate scores: only search shorter mates than the one found up to this point
        if (init_score > SCORE_MATE_FOUND)
            alpha = init_score - 1;

        data.clear_pv();
        Score final_score = negamax<ROOT>(position, depth, alpha, beta, data);

        // Increase window in the failed side exponentially
        while ((final_score <= alpha || final_score >= beta) && !(depth > 1 && timeout()))
        {
            bool upperbound = final_score <= alpha;
            Score bound = upperbound ? alpha : beta;
            if (upperbound)
                l_window *= 2;
            else
                r_window *= 2;

            // Output some information
            if (data.thread() == 0 && time_elapsed() > 3000)
            {
                std::cout << "info";
                std::cout << " depth " << static_cast<int>(depth);
                if (is_mate(bound))
                    std::cout << " score mate " << mate_in(bound);
                else
                    std::cout << " score cp " << bound;
                std::cout << (upperbound ? " upperbound" : " lowerbound") << std::endl;
            }

            // Increase window (without overflowing)
            alpha = std::max(((init_score - l_window) > init_score) ? (-SCORE_INFINITE) : (init_score - l_window), -SCORE_INFINITE);
            beta  = std::min(((init_score + r_window) < init_score) ? (+SCORE_INFINITE) : (init_score + r_window), +SCORE_INFINITE);

            // Open windows for large values
            if (alpha < 1000)
                alpha = -SCORE_INFINITE;
            if (beta > 1000)
                beta = SCORE_INFINITE;

            // Repeat search
            data.clear_pv();
            final_score = negamax<ROOT>(position, depth, alpha, beta, data);
        }

        return final_score;
    }



    Score iter_deepening(Position& position, SearchData& data)
    {
        // Maximum PV lines
        int maxPV = std::min(position.generate_moves(MoveGenType::LEGAL).length(), Parameters::multiPV);

        Score score = -SCORE_INFINITE;
        MoveStack pv_stack(maxPV);
        Turn turn = position.get_turn();
        Color color = turn_to_color(turn);
        std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();
        bool main_thread = data.thread() == 0;
        if (main_thread)
        {
            multiPV_roots.resize(maxPV);
            multiPV_scores.resize(maxPV);
            multiPV_lists.resize(maxPV);
        }

        bool time_management = Parameters::limits.time[turn] >= 0;

        // Check aborted search
        auto moves = position.generate_moves(MoveGenType::LEGAL);
        if (moves.length() == 0)
        {
            if (main_thread)
            {
                // Even in this case, output a score and bestmove
                std::cout << "info depth 0 score " << (position.is_check() ? "mate 0" : "cp 0") << std::endl;
                std::cout << "bestmove " << MOVE_NULL << std::endl;
                
                 // Stop the search
                std::unique_lock<std::mutex> lock(mutex);
                status = ThreadStatus::WAITING;
            }
            return SCORE_NONE;
        }

        Move bestmove = MOVE_NULL;
        Move pondermove = MOVE_NULL;

        // Iterative deepening
        data.histories().clear();
        for (int iDepth = 1;
             iDepth < NUM_MAX_DEPTH && (iDepth <= Parameters::limits.depth || Parameters::ponder);
             iDepth++)
        {
            // Set depth timer
            std::chrono::steady_clock::time_point depth_begin = std::chrono::steady_clock::now();

            // Reset counters and multiPV data
            if (main_thread)
            {
                //nodes_searched = 0;
                data.seldepth() = 0;
                for (int iPv = 0; iPv < maxPV; iPv++)
                {
                    multiPV_roots[iPv] = MOVE_NULL;
                    multiPV_lists[iPv] = pv_stack.list(iPv);
                }
            }

            // MultiPV loop
            for (int iPv = 0; iPv < maxPV; iPv++)
            {
                // Carry the aspirated search
                Score curr_score = aspiration_search(position, score, iDepth + data.thread() / 2, color, data);

                // Timeout?
                if (iDepth > 1 && timeout())
                    break;

                // Store depth results
                data.accept_pv();
                if (main_thread)
                {
                    get_pv(position, data, multiPV_lists[iPv]);
                    multiPV_scores[iPv] = curr_score;
                    multiPV_roots[iPv] = *multiPV_lists[iPv].begin();
                }
            }

            // Timeout?
            if (iDepth > 1 && timeout())
                break;

            // Depth timer
            std::chrono::steady_clock::time_point depth_end = std::chrono::steady_clock::now();
            double time_depth = std::chrono::duration_cast<std::chrono::nanoseconds>(depth_end - depth_begin).count() / 1e9;
            double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(depth_end - begin_time).count() / 1e9;

            // Output information
            if (main_thread)
            {
                for (int iPv = 0; iPv < maxPV; iPv++)
                {
                    std::cout << "info";
                    std::cout << " depth " << iDepth;
                    std::cout << " seldepth " << static_cast<int>(data.seldepth());
                    std::cout << " multipv " << iPv + 1;
                    if (is_mate(multiPV_scores[iPv]))
                        std::cout << " score mate " << mate_in(multiPV_scores[iPv]);
                    else
                        std::cout << " score cp " << multiPV_scores[iPv];
                    std::cout << " nodes " << nodes_searched;
                    std::cout << " nps " << static_cast<int>(nodes_searched / elapsed);
                    std::cout << " hashfull " << ttable.hashfull();
                    std::cout << " time " << std::max(1, static_cast<int>(elapsed * 1000));
                    std::cout << " pv " << multiPV_lists[iPv];
                    std::cout << std::endl;
                }
            }

            // Update best and ponder moves
            auto best_pv = multiPV_lists[0];
            auto move_ptr = best_pv.begin();
            bestmove = *move_ptr;
            pondermove = (best_pv.length() > 1) ? *(move_ptr + 1) : MOVE_NULL;

            // Additional time stopping conditions
            score = multiPV_scores[0];
            if (main_thread &&
                time_management &&
                !Parameters::limits.ponder &&
                !Parameters::limits.infinite)
            {
                double remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - std::chrono::steady_clock::now()).count() / 1e9;
                // Best move mate or do we expect not to have time for one more iteration?
                if (abs(score) > SCORE_MATE_FOUND ||
                    (remaining > 0 && remaining < time_depth * 1.5))
                {
                    break;
                }
            }

            // Stopping condition for mate search
            if (Parameters::limits.mate >= 1 &&
                is_mate(score) &&
                mate_in(score) <= Parameters::limits.mate)
                break;
        }

        // Output bestmove
        if (main_thread)
        {
            std::cout << "bestmove " << bestmove;
            if (pondermove != MOVE_NULL)
                std::cout << " ponder " << pondermove;
            std::cout << std::endl;

            // Stop the search
            std::unique_lock<std::mutex> lock(mutex);
            status = ThreadStatus::WAITING;
        }

        return score;
    }



    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data)
    {
        // Node data
        constexpr bool PvNode = ST != NON_PV;
        const bool RootSearch = ST == ROOT;
        const bool HasExcludedMove = data.excluded_move() != MOVE_NULL;
        const bool IsCheck = position.is_check();
        const Turn Turn = position.get_turn();
        const Depth Ply = data.ply();
        HistoryContext hist(data.histories(), position.board(), Ply, depth, data.last_move());

        if (PvNode)
        {
            // Update seldepth and clear PV
            *(data.pv()) = MOVE_NULL;
            data.seldepth() = std::max(data.seldepth(), Ply);
        }

        // Timeout?
        if (depth > 1 && timeout())
            return SCORE_NONE;

        // Mate distance prunning: don't bother searching if we are deeper than the shortest mate up to this point
        if (!RootSearch)
        {
            alpha = std::max(alpha, static_cast<Score>(-SCORE_MATE + Ply));
            beta  = std::min(beta,  static_cast<Score>( SCORE_MATE - Ply + 1));
            if (alpha >= beta)
                return alpha;
        }

        // Dive into quiescence at leaf nodes
        if (depth <= 0)
            return quiescence<ST>(position, alpha, beta, data);

        // Early check for draw or maximum depth reached
        if (position.is_draw(!RootSearch) ||
            Ply >= NUM_MAX_PLY)
            return SCORE_DRAW;

        // TT lookup
        Score alpha_init = alpha;
        Depth tt_depth = 0;
        Move tt_move = MOVE_NULL;
        Score tt_score = SCORE_NONE;
        Score tt_static_eval = SCORE_NONE;
        TranspositionEntry* entry = nullptr;
        EntryType tt_type = EntryType::EXACT;
        Hash hash = HasExcludedMove ? position.hash() ^ Zobrist::get_move_hash(data.excluded_move()) : position.hash();
        bool tt_hit = ttable.query(hash, &entry);
        if (tt_hit)
        {
            tt_type = entry->type();
            tt_depth = entry->depth();
            tt_score = score_from_tt(entry->score(), Ply);
            tt_move = entry->hash_move();
            tt_static_eval = entry->static_eval();

            // TT cutoff in non-PV nodes
            if (!PvNode && tt_depth >= depth &&
                ((tt_type == EntryType::EXACT) ||
                 (tt_type == EntryType::UPPER_BOUND && tt_score <= alpha) ||
                 (tt_type == EntryType::LOWER_BOUND && tt_score >= beta)))
            {
                // Update histories for quiet TT moves
                if (tt_move != MOVE_NULL && !tt_move.is_capture() && !tt_move.is_promotion())
                {
                    PieceType piece = static_cast<PieceType>(position.board().get_piece_at(tt_move.from()));
                    if (tt_score >= beta)
                        hist.fail_high(tt_move);
                    else
                        hist.add_bonus(tt_move, -depth);
                }

                // Do not cutoff when we are approaching the 50 move rule
                if (position.board().half_move_clock() < 90)
                    return tt_score;
            }
        }

        // Position static evaluation (when not in check)
        Score static_eval = SCORE_NONE;
        if (!IsCheck)
        {
            // We don't recompute static eval if
            // 1. We have a valid TT hit
            // 2. Previous move was a null-move
            if (tt_hit && tt_static_eval != SCORE_NONE)
                static_eval = tt_static_eval;
            else if (data.last_move() == MOVE_NULL && Ply > 1)
                static_eval = -data.previous(1)->static_eval();
            else
                static_eval = turn_to_color(Turn) * evaluation(position);
        }
        data.static_eval() = static_eval;

        // Can we use the TT value for a better static evaluation?
        if (tt_hit && tt_score != SCORE_NONE &&
            ((tt_type == EntryType::EXACT) ||
             (tt_type == EntryType::LOWER_BOUND && tt_score > static_eval) ||
             (tt_type == EntryType::UPPER_BOUND && tt_score < static_eval)))
            static_eval = tt_score;

        // Futility pruning
        if (!PvNode && depth < 5 && !IsCheck && !is_mate(static_eval))
        {
            Score margin = 200 * depth;
            if (static_eval - margin >= beta)
                return static_eval;
        }

        // Null move pruning
        if (!PvNode && !IsCheck && !HasExcludedMove &&
            static_eval >= beta &&
            data.last_move() != MOVE_NULL &&
            position.board().non_pawn_material(Turn))
        {
            int reduction = 3 + (static_eval - beta) / 200;
            Depth new_depth = reduce(depth, 1 + reduction);
            SearchData curr_data = data.next(MOVE_NULL) | REDUCED;
            position.make_null_move();
            nodes_searched++;
            data.nodes_searched()++;
            Score null = -negamax<NON_PV>(position, new_depth, -beta, -beta + 1, curr_data);
            position.unmake_null_move();
            if (null >= beta)
                return null < SCORE_MATE_FOUND ? null : beta;
        }

        // TT-based reduction idea
        if (PvNode && !IsCheck && depth >= 6 && entry == nullptr)
            depth -= 2;

        // Regular move search
        Move move;
        int n_moves = 0;
        int move_number = 0;
        Move best_move = MOVE_NULL;
        Score best_score = -SCORE_INFINITE;
        Move quiet_list[NUM_MAX_MOVES];
        MoveList quiets_searched(quiet_list);
        Move hash_move = (data.in_pv() && data.pv_move() != MOVE_NULL) ? data.pv_move() : tt_move;
        MoveOrder orderer = MoveOrder(position, hash_move, hist);
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            n_moves++;
            if (!move.is_capture() && !move.is_promotion())
            {
                quiets_searched.push(move);
                move_number++;
            }

            // Skip excluded moves
            if (move == data.excluded_move())
                continue;

            // New search parameters
            Depth curr_depth = depth;
            SearchData curr_data = data.next(move);
            uint64_t move_nodes = data.nodes_searched();

            // In multiPV mode do not search previous PV root moves
            if (RootSearch && Parameters::multiPV > 1)
            {
                bool found = false;
                for (auto root_moves : multiPV_roots)
                    found |= (root_moves == move);
                if (found)
                    continue;
            }

            // Only search for selected root moves if specified
            if (RootSearch && Parameters::limits.searchmoves.size() != 0)
            {
                bool found = false;
                for (auto root_moves : Parameters::limits.searchmoves)
                    found |= (root_moves == move);
                if (!found)
                    continue;
            }

            // Output some information during search
            if (RootSearch && data.thread() == 0 &&
                time_elapsed() > 3000)
                std::cout << "info depth " << static_cast<int>(depth)
                          << " currmove " << move.to_uci()
                          << " currmovenumber " << n_moves << std::endl;

            // Shallow depth prunings
            if (!RootSearch && position.board().non_pawn_material(Turn) && !IsCheck && best_score > -SCORE_MATE_FOUND)
            {
                if (move.is_capture() || move.is_promotion())
                {
                    if (depth < 7 && position.board().see(move, -200 * depth) < 0)
                        continue;
                }
                else
                {
                    if (depth < 7 && n_moves > 3 + depth * depth)
                        continue;

                    if (depth < 5 && orderer.quiet_score(move) < -3000 * (depth - 1))
                        continue;

                    if (depth < 7 && position.board().see(move, -20 * (depth + (int)depth * depth)) < 0)
                        continue;
                }
            }

            // Singular extensions: when the stored TT value fails high, we carry a reduced search on the remaining moves
            // If all moves fail low then we extend the TT move
            if (!RootSearch && 
                tt_hit &&
                depth > 8 &&
                !HasExcludedMove &&
                !IsCheck &&
                move == tt_move &&
                tt_depth >= depth - 3 &&
                tt_type == EntryType::LOWER_BOUND &&
                !is_mate(tt_score) &&
                data.extensions() < 3)
            {
                Value singularBeta = tt_score - 2 * depth;
                Depth singularDepth = (depth - 1) / 2;

                // Search with the move excluded
                curr_data.exclude(move);
                curr_data |= REDUCED;
                Score score = negamax<NON_PV>(position, singularDepth, singularBeta - 1, singularBeta, curr_data);
                curr_data ^= REDUCED;
                curr_data.exclude(MOVE_NULL);

                if (score < singularBeta)
                {
                    // TT move is singular, we are extending it
                    curr_data |= EXTENDED;
                    curr_depth++;
                }
                else if (singularBeta >= beta)
                {
                    // Multi-cut pruning: assuming our TT move fails high, at least one more move also fails high
                    // So we can probably safely prune the entire tree
                    return singularBeta;
                }
            }

            // Make the move
            Score score;
            bool captureOrPromotion = move.is_capture() || move.is_promotion();
            PieceType piece = static_cast<PieceType>(position.board().get_piece_at(move.from()));
            position.make_move(move);
            nodes_searched++;
            data.nodes_searched()++;

            // Late move reductions
            bool do_full_search = true;
            bool didLMR = false;
            if (depth > 4 &&
                move_number > 3 &&
                (!PvNode || !captureOrPromotion) &&
                data.thread() % 3 < 2)
            {
                didLMR = true;
                int reduction = 3 + (move_number - 4) / 8 - captureOrPromotion - PvNode;
                Depth new_depth = reduce(depth, 1 + reduction);

                // Reduced depth search
                curr_data |= REDUCED;
                score = -negamax<NON_PV>(position, new_depth, -alpha - 1, -alpha, curr_data);
                curr_data ^= REDUCED;

                // Only carry a full search if this reduced move fails high
                do_full_search = score >= alpha;
            }

            // Check extensions
            if (IsCheck && data.extensions() < 3 && depth < 4)
            {
                curr_data |= EXTENDED;
                curr_depth++;
            }

            // PVS
            if (do_full_search)
            {
                if (PvNode && n_moves == 1)
                {
                    score = -negamax<PV>(position, curr_depth - 1, -beta, -alpha, curr_data);

                    // Return failed aspirated search immediately
                    if (RootSearch && (score <= alpha || score >= beta))
                    {
                        position.unmake_move();
                        return score;
                    }
                }
                else
                {
                    // Regular non-PV node search
                    score = -negamax<NON_PV>(position, curr_depth - 1, -alpha - 1, -alpha, curr_data);
                    // Redo a PV node search if move not refuted
                    if (PvNode && score > alpha && score < beta)
                    {
                        // But before add a bonus to the move
                        hist.add_bonus(move, Ply, piece, depth);
                        score = -negamax<PV>(position, curr_depth - 1, -beta, -alpha, curr_data);
                    }
                }
            }

            // Unmake the move
            position.unmake_move();

            // Timeout?
            if (depth > 2 && timeout())
                return SCORE_NONE;

            // Update histories after passed LMR
            if (didLMR && do_full_search)
            {
                int bonus = score > best_score ? depth : -depth;
                hist.add_bonus(move, Ply, piece, bonus);
            }

            // New best move
            if (score > best_score)
            {
                best_score = score;
                best_move = move;
                alpha = std::max(alpha, score);

                // Update PV when we have a bestmove
                if (PvNode)
                    data.update_pv(best_move, curr_data.pv());
            }

            // Update low-ply history
            move_nodes = data.nodes_searched() - move_nodes;
            if (Ply < NUM_LOW_PLY)
                hist.update_low_ply(move, move_nodes / 10000);

            // Pruning
            if (alpha >= beta)
            {
                if (!move.is_capture())
                    hist.fail_high(move);
                break;
            }
        }

        // Update quiet histories (penalise searched moves if some move raised alpha)
        if (best_score >= alpha)
            for (auto move : quiets_searched)
                if (move != best_move)
                    hist.add_bonus(move, -depth * depth / 4);

        // Check for game end
        if (n_moves == 0)
        {
            // Checkmate or stalemate?
            if (position.is_check())
                best_score = -SCORE_MATE + Ply;
            else
                best_score = SCORE_DRAW;
        }

        // TT store (except at root in non-main threads)
        if (!(RootSearch && data.thread() != 0))
        {
            Hash hash = HasExcludedMove ? position.hash() ^ Zobrist::get_move_hash(data.excluded_move()) : position.hash();
            EntryType type = best_score >= beta                  ? EntryType::LOWER_BOUND
                           : (PvNode && best_score > alpha_init) ? EntryType::EXACT
                           :                                       EntryType::UPPER_BOUND;
            ttable.store(TranspositionEntry(hash, depth, score_to_tt(best_score, Ply), best_move, type, data.static_eval()), RootSearch);
        }

        return best_score;
    }



    template<SearchType ST>
    Score quiescence(Position& position, Score alpha, Score beta, SearchData& data)
    {
        constexpr bool PvNode = ST == PV;
        const bool IsCheck = position.is_check();
        const Turn Turn = position.get_turn();
        const Depth Ply = data.ply();
        HistoryContext hist(data.histories(), position.board(), Ply, 0, data.last_move());

        if (PvNode)
        {
            // Update seldepth and clear PV
            *(data.pv()) = MOVE_NULL;
            data.seldepth() = std::max(data.seldepth(), Ply);
        }

        // Early check for draw or maximum depth reached
        if (position.is_draw(true) ||
            Ply >= NUM_MAX_PLY)
            return SCORE_DRAW;

        // Mate distance pruning: don't bother searching if we are deeper than the shortest mate up to this point
        alpha = std::max(alpha, static_cast<Score>(-SCORE_MATE + Ply));
        beta = std::min(beta, static_cast<Score>(SCORE_MATE - Ply + 1));
        if (alpha >= beta)
            return alpha;

        // TT lookup
        Score alpha_init = alpha;
        Move tt_move = MOVE_NULL;
        Score tt_score = SCORE_NONE;
        Score tt_static_eval = SCORE_NONE;
        TranspositionEntry* entry = nullptr;
        EntryType tt_type = EntryType::EXACT;
        bool tt_hit = ttable.query(position.hash(), &entry);
        if (tt_hit)
        {
            tt_type = entry->type();
            tt_score = score_from_tt(entry->score(), Ply);
            tt_move = entry->hash_move();
            tt_static_eval = entry->static_eval();

            // In quiescence ensure the tt_move is a capture in non-check positions
            if (!IsCheck && !tt_move.is_capture())
                tt_move = MOVE_NULL;

            // TT cutoff in non-PV nodes
            if (!PvNode)
            {
                if ((tt_type == EntryType::EXACT) ||
                    (tt_type == EntryType::UPPER_BOUND && tt_score <= alpha) ||
                    (tt_type == EntryType::LOWER_BOUND && tt_score >= beta))
                    return tt_score;
            }
        }

        // Position static evaluation (when not in check)
        Score static_eval = SCORE_NONE;
        Score best_score = -SCORE_INFINITE;
        if (!IsCheck)
        {
            // Don't recompute static eval if we have a valid TT hit
            if (tt_hit && tt_static_eval != SCORE_NONE)
                static_eval = tt_static_eval;
            else
                static_eval = turn_to_color(Turn) * evaluation(position);
            best_score = static_eval;

            // Can we use the TT value for a better static evaluation?
            if (tt_hit && abs(tt_score) < SCORE_MATE_FOUND &&
                ((tt_type == EntryType::EXACT) ||
                 (tt_type == EntryType::LOWER_BOUND && tt_score > static_eval) ||
                 (tt_type == EntryType::UPPER_BOUND && tt_score < static_eval)))
                best_score = tt_score;

            // Alpha-beta pruning on stand pat
            alpha = std::max(alpha, best_score);
            if (alpha >= beta)
                return alpha;
        }

        // Search
        Move move;
        int n_moves = 0;
        Move best_move = MOVE_NULL;
        MoveOrder orderer = MoveOrder(position, tt_move, hist, true);
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            n_moves++;

            // Only search captures with positive SEE
            if (!IsCheck && position.board().see(move) < 0)
                continue;

            // PVS
            Score score;
            position.make_move(move);
            nodes_searched++;
            data.nodes_searched()++;
            auto curr_data = data.next(move);
            if (PvNode && best_move == MOVE_NULL)
            {
                score = -quiescence<PV>(position, -beta, -alpha, curr_data);
            }
            else
            {
                // Regular non-PV node search
                score = -quiescence<NON_PV>(position, -alpha - 1, -alpha, curr_data);
                // Redo a PV node search if move not refuted
                if (PvNode && score > alpha && score < beta)
                    score = -quiescence<PV>(position, -beta, -alpha, curr_data);
            }
            position.unmake_move();

            // New best value
            if (score > best_score)
            {
                best_score = score;
                best_move = move;
                alpha = std::max(alpha, best_score);

                // Update PV in PvNodes
                if (PvNode)
                    data.update_pv(best_move, curr_data.pv());

                // Pruning
                if (alpha >= beta)
                    break;
            }
        }

        // Checkmate?
        if (n_moves == 0 && IsCheck)
            return -SCORE_MATE + Ply;

        // TT store
        EntryType type = best_score >= beta                  ? EntryType::LOWER_BOUND
                       : (PvNode && best_score > alpha_init) ? EntryType::EXACT
                       :                                       EntryType::UPPER_BOUND;
        ttable.store(TranspositionEntry(position.hash(), 0, score_to_tt(best_score, Ply), best_move, type, static_eval));

        return best_score;
    }


    bool legality_tests(Position& position, MoveList& move_list)
    {
        bool final = true;
        // Legality check
        for (auto move : move_list)
            if (!position.board().legal(move))
            {
                std::cout << "Bad illegal move " << move.to_uci() << " (" << move.to_int() << ") in " << position.board().to_fen() << std::endl;
                final = false;
            }

        // Illegality check: first count number of legal moves
        int result = 0;
        for (uint16_t number = 0; number < UINT16_MAX; number++)
            if (position.board().legal(Move::from_int(number)))
                result++;
        // Something is wrong, find the bad legals
        if (result != move_list.length())
        {
            std::cout << result << " vs " << move_list.length() << std::endl;
            for (uint16_t number = 0; number < UINT16_MAX; number++)
            {
                Move move = Move::from_int(number);
                if (position.board().legal(move) && !move_list.contains(move))
                    std::cout << "Bad legal move " << move.to_uci() << " (" << move.to_int() << ") in " << position.board().to_fen() << std::endl;
            }
            final = false;
        }
        return final;
    }
}
