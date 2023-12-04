#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include "MoveOrder.hpp"
#include "Search.hpp"
#include "Evaluation.hpp"
#include "UCI.hpp"
#include "Zobrist.hpp"
#include "Thread.hpp"
#include "syzygy/syzygy.hpp"
#include <atomic>
#include <chrono>
#include <vector>

namespace Search
{
    Timer::Timer()
    {
        m_start = std::chrono::steady_clock::now();
    }

    double Timer::diff(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b)
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(a - b).count() / 1e9;
    }

    double Timer::elapsed() const
    {
        return diff(std::chrono::steady_clock::now(), m_start);
    }

    std::chrono::steady_clock::time_point Timer::begin() const
    {
        return m_start;
    }



    SearchTime::SearchTime() noexcept
        : m_managed(false),
          m_end_time(std::chrono::steady_clock::now())
    {}

    void SearchTime::init(const Timer& timer, bool ponder)
    {
        m_timer = timer;
        m_managed = false;
        m_pondering.store(ponder);
    }

    void SearchTime::init(const Timer& timer, uint64_t movetime_ms, bool ponder)
    {
        m_timer = timer;
        m_managed = true;
        m_movetime_ms = movetime_ms;
        m_optimum = std::numeric_limits<double>::infinity();
        m_pondering.store(ponder);
        m_end_time.store(m_timer.begin() + std::chrono::milliseconds(movetime_ms));
    }

    void SearchTime::init(const Timer& timer, uint64_t movetime_ms, uint64_t optimum_ms, bool ponder)
    {
        m_timer = timer;
        m_managed = true;
        m_movetime_ms = movetime_ms;
        m_optimum = optimum_ms / 1000.0;
        m_pondering.store(ponder);
        m_end_time.store(m_timer.begin() + std::chrono::milliseconds(movetime_ms));
    }

    void SearchTime::ponderhit()
    {
        m_pondering.store(false);
        if (m_managed)
            m_end_time.store(std::chrono::steady_clock::now() + std::chrono::milliseconds(m_movetime_ms));
    }

    bool SearchTime::pondering() const
    {
        return m_pondering.load(std::memory_order_relaxed);
    }

    double SearchTime::elapsed() const
    {
        return m_timer.elapsed();
    }

    double SearchTime::optimum() const
    {
        return m_optimum;
    }

    double SearchTime::remaining() const
    {
        // Return infinite remaining time if pondering or if search is not time managed
        if (pondering() || !time_management())
            return std::numeric_limits<double>::infinity();

        // Remaining time is always the difference between now and the end time
        return Timer::diff(m_end_time.load(std::memory_order_relaxed),
                           std::chrono::steady_clock::now());
    }
    
    bool SearchTime::time_management() const
    {
        return m_managed;
    }



    MultiPVData::MultiPVData()
        : depth(0),
          seldepth(0),
          search_score(-SCORE_NONE),
          search_bound(BoundType::NO_BOUND),
          tb_score(-SCORE_NONE)
    {
    }

    Score MultiPVData::score() const
    {
        // Use search score if:
        // 1. We have no TB info
        // 2. We have found a forced mate
        if (!Syzygy::Root.in_tb() ||
            (is_mate(this->search_score) && this->search_bound == BoundType::EXACT))
            return this->search_score;

        return this->tb_score;
    }

    BoundType MultiPVData::bound() const
    {
        // Use search bound if:
        // 1. We have no TB info
        // 2. We have found a forced mate
        if (!Syzygy::Root.in_tb() ||
            (is_mate(this->search_score) && this->search_bound == BoundType::EXACT))
            return this->search_bound;

        // Choose bound based on the type of TB score
        return this->tb_score ==  SCORE_DRAW         ? BoundType::EXACT        // Draw
             : this->tb_score >=  SCORE_TB_WIN_FOUND ? BoundType::LOWER_BOUND  // Win
             : this->tb_score <= -SCORE_TB_WIN_FOUND ? BoundType::UPPER_BOUND  // Loss
             : this->tb_score >   SCORE_DRAW         ? BoundType::UPPER_BOUND  // Cursed win
             :                                         BoundType::LOWER_BOUND; // Blessed loss
    }

    void MultiPVData::write_pv(int index, uint64_t nodes, uint64_t tb_hits, double elapsed) const
    {
        // Don't write if PV line is incomplete
        if (search_bound == BoundType::NO_BOUND)
            return;

        std::cout << "info";
        std::cout << " depth "    << static_cast<int>(depth);
        std::cout << " seldepth " << static_cast<int>(seldepth);
        std::cout << " multipv "  << index + 1;

        // Score
        if (is_mate(this->score()))
            std::cout << " score mate " << mate_in(this->score());
        else
            std::cout << " score cp " << 100 * int(this->score()) / PawnValue.endgame();

        // Score bound (if any)
        if (this->bound() != BoundType::EXACT)
            std::cout << (this->bound() == BoundType::LOWER_BOUND ? " lowerbound" : " upperbound");

        // Nodes, nps, hashful and timing
        std::cout << " nodes "    << nodes;
        std::cout << " nps "      << static_cast<int>(nodes / elapsed);
        std::cout << " hashfull " << ttable.hashfull();
        std::cout << " tbhits "   << tb_hits;
        std::cout << " time "     << std::max(1, static_cast<int>(elapsed * 1000));

        // Pv line
        const Move* m = pv;
        std::cout << " pv " << *(m++);
        while (*m != MOVE_NULL)
            std::cout << " " << (*m++);
        
        std::cout << std::endl;
    }



    SearchData::SearchData(Thread& thread)
        : m_ply(0), m_prev(nullptr), m_thread(thread), m_move(MOVE_NULL),
          m_pv(thread.m_pv.pv), m_prev_pv(thread.m_pv.prev_pv), m_isPv(true), 
          seldepth(thread.m_seldepth), static_eval(SCORE_NONE), excluded_move(MOVE_NULL),
          histories(thread.m_histories)
    {}

    int SearchData::ply() const { return m_ply; }
    bool SearchData::in_pv() const { return m_isPv; }
    Move SearchData::last_move() const { return m_move; }
    Move* SearchData::pv() { return m_pv; }
    Move* SearchData::prev_pv() { return m_prev_pv; }
    Move SearchData::pv_move() { return m_isPv ? m_prev_pv[m_ply] : MOVE_NULL; }
    Thread& SearchData::thread() const { return m_thread; }
    uint64_t SearchData::nodes_searched() const { return m_thread.m_nodes_searched.load(std::memory_order_relaxed); }

    SearchData SearchData::next(Move move) const
    {
        SearchData result = *this;
        result.m_prev = this;
        result.m_move = move;
        // Other parameters
        result.m_ply++;
        result.static_eval = SCORE_NONE;
        result.excluded_move = MOVE_NULL;
        // New pv location
        result.m_pv += PV_LENGTH - m_ply;
        // Are we still in a PV line?
        result.m_isPv = m_isPv && move == m_prev_pv[m_ply];
        // Increment searched nodes
        m_thread.m_nodes_searched.fetch_add(1, std::memory_order_relaxed);
        return result;
    }

    const SearchData* SearchData::previous(int distance) const
    {
        const SearchData* curr = this;
        for (int i = 0; i < distance; i++)
            curr = curr->m_prev;
        return curr;
    }

    void SearchData::update_pv(Move best_move, Move* new_pv)
    {
        // Set the initial bestmove
        Move* dst = m_pv;
        *(dst++) = best_move;

        // Loop over remaining PV
        while (new_pv && *new_pv != MOVE_NULL)
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


    void copy_pv(Move* src, Move* dst)
    {
        while (*src != MOVE_NULL)
            *(dst++) = *(src++);

        // Set last entry as null move as a stop condition
        *dst = MOVE_NULL;
    }



    Score aspiration_search(Position& position, MultiPVData& pv, Depth depth, SearchData& data)
    {
        Thread& thread = data.thread();

        // Initial windows
        Score init_score = pv.search_score;
        Score window = 5 + 200 / depth;
        Score alpha = (depth <= 4) ? (-SCORE_INFINITE) : std::max(-SCORE_INFINITE, (init_score - window));
        Score beta  = (depth <= 4) ? ( SCORE_INFINITE) : std::min(+SCORE_INFINITE, (init_score + window));

        Score score = -SCORE_INFINITE;
        while (true)
        {
            // Open windows for large values
            if (alpha < -1000)
                alpha = -SCORE_INFINITE;
            if (beta > 1000)
                beta = SCORE_INFINITE;

            // Clear Pv output array and fetch previous Pv line for move ordering
            data.clear_pv();
            data.seldepth = 0;
            copy_pv(pv.pv, data.prev_pv());

            // Do the search
            score = negamax<ROOT>(position, depth, alpha, beta, data, false);

            // Check for timeout: search results cannot be trusted
            if (thread.timeout() && depth > 1)
                return score;

            // Inject TB data, if any
            if (Syzygy::Root.in_tb())
                pv.tb_score = Syzygy::Root.move_score(*data.pv());

            // Store results for this PV line
            pv.depth = depth;
            pv.search_score = score;
            pv.seldepth = data.seldepth;
            pv.search_bound = score <= alpha ? BoundType::UPPER_BOUND
                            : score >= beta  ? BoundType::LOWER_BOUND
                            :                  BoundType::EXACT;
            copy_pv(data.pv(), pv.pv);

            // We can exit if this score is exact
            if (pv.search_bound == BoundType::EXACT)
                return score;

            // Output failed search after some time
            if (thread.is_main() && thread.time().elapsed() > 3 && UCI::Options::MultiPV == 1)
                thread.output_pvs();

            // Increase window in the failed side exponentially
            window *= 2;
            if (score <= alpha)
                alpha = std::max(score - window, -SCORE_INFINITE);
            else
                beta = std::min(score + window, +SCORE_INFINITE);
        }

        return score;
    }



    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data, bool cut_node)
    {
        // Node data
        constexpr bool PvNode = ST != NON_PV;
        constexpr bool RootSearch = ST == ROOT;
        const bool HasExcludedMove = data.excluded_move != MOVE_NULL;
        const bool InCheck = position.in_check();
        const Turn Turn = position.get_turn();
        const Depth Ply = data.ply();
        CurrentHistory history = data.histories.get(position);

        if (PvNode)
        {
            // Update seldepth and clear PV
            data.update_pv(MOVE_NULL, nullptr);
            data.seldepth = std::max(data.seldepth, Ply);
        }

        // Timeout?
        if (depth > 3 && data.thread().timeout())
            return SCORE_NONE;

        // Mate distance pruning: don't bother searching if we are deeper than the shortest mate up to this point
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
        Hash hash = HasExcludedMove ? position.hash() ^ Zobrist::get_move_hash(data.excluded_move) : position.hash();
        bool tt_hit = ttable.query(hash, &entry);
        if (tt_hit)
        {
            tt_type = entry->type();
            tt_depth = entry->depth();
            tt_score = score_from_tt(entry->score(), Ply, position.board().half_move_clock());
            tt_move = entry->hash_move();
            tt_static_eval = entry->static_eval();

            // TT cutoff in non-PV nodes
            if (!PvNode && tt_depth >= depth &&
                ((tt_type == EntryType::EXACT) ||
                 (tt_type == EntryType::UPPER_BOUND && tt_score <= alpha) ||
                 (tt_type == EntryType::LOWER_BOUND && tt_score >= beta)))
            {
                // Update histories for quiet TT moves
                if (tt_move != MOVE_NULL && !tt_move.is_capture() && tt_score >= beta && position.board().legal(tt_move))
                    history.bestmove(tt_move, position.board().get_piece_at(tt_move.from()), depth);

                // Do not cutoff when we are approaching the 50 move rule
                if (position.board().half_move_clock() < 90)
                    return tt_score;
            }
        }

        // Tablebases probe
        Score best_score = -SCORE_INFINITE;
        Score max_score = SCORE_INFINITE;
        if (!RootSearch &&
            !HasExcludedMove &&
            Syzygy::Cardinality > 0 &&
            Syzygy::Cardinality >= position.board().get_pieces().count() &&
            depth >= UCI::Options::TB_ProbeDepth &&
            position.board().half_move_clock() == 0 &&
            !position.board().can_castle())
        {
            Syzygy::WDL wdl = Syzygy::probe_wdl(position.board());
            if (wdl != Syzygy::WDL_NONE)
            {
                // Derive the TB score and bound
                data.thread().tb_hit();
                Score score = Syzygy::score_from_wdl(wdl, Ply);
                EntryType bound = Syzygy::bound_from_wdl(wdl);

                // Prune and cache TB score in TT
                if ( bound == EntryType::EXACT ||
                    (bound == EntryType::LOWER_BOUND && score >= beta) ||
                    (bound == EntryType::UPPER_BOUND && score <= alpha))
                {
                    ttable.store(hash, depth, score_to_tt(score, Ply), MOVE_NULL, bound, SCORE_NONE);
                    return score;
                }

                if (PvNode)
                {
                    if (bound == EntryType::LOWER_BOUND)
                    {
                        best_score = score;
                        alpha = std::max(score, alpha);
                    }
                    else
                    {
                        max_score = score;
                    }
                }
            }
        }

        // Position static evaluation (when not in check)
        Score static_eval = SCORE_NONE;
        if (!InCheck)
        {
            // We don't recompute static eval if we have a valid TT hit
            if (tt_hit && tt_static_eval != SCORE_NONE)
                static_eval = tt_static_eval;
            else
                static_eval = turn_to_color(Turn) * Evaluation::evaluation(position.board());
        }
        data.static_eval = static_eval;

        // Can we use the TT value for a better static evaluation?
        if (tt_hit && tt_score != SCORE_NONE &&
            ((tt_type == EntryType::EXACT) ||
             (tt_type == EntryType::LOWER_BOUND && tt_score > static_eval) ||
             (tt_type == EntryType::UPPER_BOUND && tt_score < static_eval)))
            static_eval = tt_score;

        bool improving = !InCheck && Ply >= 2 && data.previous(2) && (data.static_eval > data.previous(2)->static_eval);

        // Futility pruning
        if (!PvNode && depth < 9 && !InCheck && abs(static_eval) < SCORE_TB_WIN_FOUND)
        {
            Score margin = 150 * (depth - improving);
            if (static_eval - margin >= beta)
                return static_eval;
        }

        // Null move pruning
        if (!PvNode && !InCheck && !HasExcludedMove &&
            static_eval >= beta &&
            static_eval >= data.static_eval &&
            data.static_eval >= beta - 20 * depth - 40 * improving + 100 &&
            data.last_move() != MOVE_NULL &&
            position.board().non_pawn_material(Turn))
        {
            int reduction = 4 + std::min(6, (static_eval - beta) / 200);
            Depth new_depth = std::max(0, depth - reduction);
            SearchData curr_data = data.next(MOVE_NULL);
            position.make_null_move();
            Score null = -negamax<NON_PV>(position, new_depth, -beta, -beta + 1, curr_data, !cut_node);
            position.unmake_null_move();
            if (null >= beta)
                return null < SCORE_TB_WIN_FOUND ? null : beta;
        }

        // TT-based reduction idea
        if (!RootSearch && PvNode && !InCheck && !position.board().legal(tt_move))
            depth -= 2;

        if (depth <= 0)
            return quiescence<ST>(position, alpha, beta, data);

        // Regular move search
        Move move;
        int n_moves = 0;
        MoveList quiets_searched;
        Move best_move = MOVE_NULL;
        Move hash_move = (data.in_pv() && data.pv_move() != MOVE_NULL) ? data.pv_move() : tt_move;
        MoveOrder orderer = MoveOrder(position, depth, hash_move, history);
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            // Skip excluded moves
            if (move == data.excluded_move)
                continue;

            // For the root node, only search the stored root moves
            if (RootSearch && !data.thread().is_root_move(move))
                continue;

            // New search parameters
            n_moves++;
            int extension = 0;
            Depth curr_depth = depth;

            // Output some information during search
            if (RootSearch && data.thread().is_main() &&
                data.thread().time().elapsed() > 3)
                std::cout << "info depth " << depth
                          << " currmove " << move.to_uci()
                          << " currmovenumber " << n_moves << std::endl;

            // Shallow depth pruning
            int move_score = move.is_capture() ? 0 : orderer.quiet_score(move);
            if (!RootSearch && position.board().non_pawn_material(Turn) && best_score > -SCORE_TB_WIN_FOUND)
            {
                if (move.is_capture() || move.is_promotion())
                {
                    if (depth < 10 && position.board().see(move, -140 * depth) < 0)
                        continue;
                }
                else
                {
                    if (depth < 7 && n_moves > 3 + depth * depth)
                        continue;

                    if (move_score < -100 * (depth - 1) - 50 * int(depth) * depth * depth)
                        continue;

                    if (!InCheck && depth < 12 && data.static_eval + 100 + 150 * depth + move_score / 75 < alpha)
                        continue;

                    if (depth < 10 && position.board().see(move, -10 * (depth + (int)depth * depth)) < 0)
                        continue;
                }
            }

            // Singular extensions: when the stored TT value fails high, we carry a reduced search on the remaining moves
            // If all moves fail low then we extend the TT move
            if (!RootSearch && 
                tt_hit &&
                depth > 8 &&
                !HasExcludedMove &&
                move == tt_move &&
                tt_depth >= depth - 3 &&
                (tt_type == EntryType::LOWER_BOUND || tt_type == EntryType::EXACT) &&
                !is_mate(tt_score))
            {
                Score singularBeta = tt_score - 2 * depth;
                Depth singularDepth = (depth - 1) / 2;

                // Search with the move excluded
                data.excluded_move = move;
                Score score = negamax<NON_PV>(position, singularDepth, singularBeta - 1, singularBeta, data, cut_node);
                data.excluded_move = MOVE_NULL;

                if (score < singularBeta)
                {
                    // TT move is singular, we are extending it
                    extension = 1;

                    if (!PvNode && score < singularBeta - 40 && Ply < data.thread().root_depth())
                        extension = 2;
                }
                else if (!PvNode && singularBeta >= beta)
                {
                    // Multi-cut pruning: assuming our TT move fails high, at least one more move also fails high
                    // So we can probably safely prune the entire tree
                    return singularBeta;
                }
            }

            // Make the move
            Score score = -SCORE_INFINITE;
            bool captureOrPromotion = move.is_capture() || move.is_promotion();
            PieceType piece = position.board().get_piece_at(move.from());
            position.make_move(move);

            // Check extensions
            if (position.in_check() && abs(static_eval) > 75)
                extension = 1;

            // Update depth and search data
            curr_depth = depth + extension;
            SearchData curr_data = data.next(move);

            // Late move reductions
            bool do_full_search = !(PvNode && n_moves == 1);
            bool didLMR = false;
            if (depth > 2 &&
                n_moves > 1 + 2 * RootSearch &&
                (!PvNode || !captureOrPromotion))
            {
                didLMR = true;
                int reduction = ilog2(n_moves) / 2
                              - (captureOrPromotion || PvNode)
                              - (move_score + 15000) / 30000
                              + 2 * cut_node
                              + 2;

                // Reduced depth search
                Depth new_depth = std::clamp(curr_depth - reduction - 1, 0, curr_depth - 1);
                score = -negamax<NON_PV>(position, new_depth, -alpha - 1, -alpha, curr_data, true);

                // Only carry a full search if this reduced move fails high
                do_full_search = score > alpha && reduction > 0;
            }

            // NonPv node search when LMR is skipped or fails high
            if (do_full_search)
                score = -negamax<NON_PV>(position, curr_depth - 1, -alpha - 1, -alpha, curr_data, !cut_node);

            // PvNode search at the first move in a PV node or when the nonPv search returns
            // a possibly good move
            if (PvNode && (n_moves == 1 || (score > alpha && (RootSearch || score < beta))))
            {
                score = -negamax<PV>(position, curr_depth - 1, -beta, -alpha, curr_data, false);

                // Return failed aspirated search immediately
                if (RootSearch && n_moves == 1 && (score <= alpha || score >= beta))
                {
                    position.unmake_move();
                    data.update_pv(move, curr_data.pv());
                    return score;
                }
            }

            // Unmake the move
            position.unmake_move();

            // After a timeout, the search results cannot be trusted
            if (RootSearch && data.thread().timeout() && depth > 1)
                return best_score;

            // Update histories after passed LMR
            if (didLMR && do_full_search)
            {
                int bonus = score > best_score ? hist_bonus(depth) : -hist_bonus(depth);
                history.add_bonus(move, piece, bonus);
            }

            // New best move
            if (score > best_score)
            {
                best_score = score;

                if (score > alpha)
                {
                    alpha = score;
                    best_move = move;

                    // Pruning
                    if (alpha >= beta)
                    {
                        // Track the move causing the fail-high
                        if (PvNode)
                            data.update_pv(best_move, nullptr);

                        break;
                    }

                    // Update PV when we have a new bestmove
                    if (PvNode)
                        data.update_pv(best_move, curr_data.pv());
                }
            }

            // Update list of searched moves
            if (!move.is_capture())
                quiets_searched.push(move);
        }

        // Update quiet histories
        if (best_move != MOVE_NULL && !best_move.is_capture())
        {
            // Update stats for bestmove
            history.bestmove(best_move, position.board().get_piece_at(best_move.from()), depth);

            // Penalty for any non-best quiet
            for (auto move : quiets_searched)
                if (move != best_move)
                    history.add_bonus(move, position.board().get_piece_at(move.from()), -hist_bonus(depth));
        }

        // Check for game end
        if (n_moves == 0)
        {
            // Checkmate or stalemate?
            if (position.in_check())
                best_score = -SCORE_MATE + Ply;
            else
                best_score = SCORE_DRAW;
        }

        if (PvNode)
            best_score = std::min(best_score, max_score);

        // TT store
        EntryType type = best_score >= beta                  ? EntryType::LOWER_BOUND
                       : (PvNode && best_score > alpha_init) ? EntryType::EXACT
                       :                                       EntryType::UPPER_BOUND;
        ttable.store(hash, depth, score_to_tt(best_score, Ply), best_move, type, data.static_eval);

        return best_score;
    }



    template<SearchType ST>
    Score quiescence(Position& position, Score alpha, Score beta, SearchData& data)
    {
        constexpr bool PvNode = ST == PV;
        const bool InCheck = position.in_check();
        const Turn Turn = position.get_turn();
        const Depth Ply = data.ply();

        if (PvNode)
        {
            // Update seldepth and clear PV
            data.update_pv(MOVE_NULL, nullptr);
            data.seldepth = std::max(data.seldepth, Ply);
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
            tt_score = score_from_tt(entry->score(), Ply, position.board().half_move_clock());
            tt_move = entry->hash_move();
            tt_static_eval = entry->static_eval();

            // In quiescence ensure the tt_move is a capture in non-check positions
            if (!InCheck && !tt_move.is_capture())
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
        if (!InCheck)
        {
            // Don't recompute static eval if we have a valid TT hit
            if (tt_hit && tt_static_eval != SCORE_NONE)
                static_eval = tt_static_eval;
            else
                static_eval = turn_to_color(Turn) * Evaluation::evaluation(position.board());
            best_score = static_eval;

            // Can we use the TT value for a better static evaluation?
            if (tt_hit &&
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
        CurrentHistory history = data.histories.get(position);
        MoveOrder orderer = MoveOrder(position, 0, tt_move, history, true);
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            n_moves++;

            // Only search captures with positive SEE
            if (!InCheck && position.board().see(move) < 0)
                continue;

            // PVS
            Score score;
            position.make_move(move);
            SearchData curr_data = data.next(move);
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

                if (score > alpha)
                {
                    alpha = score;
                    best_move = move;

                    // Pruning
                    if (alpha >= beta)
                    {
                        if (PvNode)
                            data.update_pv(best_move, nullptr);
                        break;
                    }

                    // Update PV in PvNodes
                    if (PvNode)
                        data.update_pv(best_move, curr_data.pv());
                }
            }
        }

        // Checkmate?
        if (n_moves == 0 && InCheck)
            return -SCORE_MATE + Ply;

        // TT store
        EntryType type = best_score >= beta                  ? EntryType::LOWER_BOUND
                       : (PvNode && best_score > alpha_init) ? EntryType::EXACT
                       :                                       EntryType::UPPER_BOUND;
        ttable.store(position.hash(), 0, score_to_tt(best_score, Ply), best_move, type, static_eval);

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
        std::size_t result = 0;
        for (uint16_t number = 0; number < UINT16_MAX; number++)
            if (position.board().legal(Move::from_int(number)))
                result++;
        // Something is wrong, find the bad legals
        if (result != move_list.size())
        {
            std::cout << result << " vs " << move_list.size() << std::endl;
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
