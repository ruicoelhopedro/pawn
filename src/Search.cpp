#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
#include "MoveOrder.hpp"
#include "Search.hpp"
#include "Evaluation.hpp"
#include <atomic>
#include <chrono>
#include <vector>

namespace Search
{
    bool thinking = false;
    int64_t nodes_searched;
    Position* base_position;
    std::vector<std::thread> threads;
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


    SearchData::SearchData(Histories& histories, int thread_id, int64_t& nodes_searched, Depth& seldepth)
        : m_histories(histories), m_ply(1), m_extensions(0), m_reductions(0),
          m_nodes_searched(nodes_searched), m_seldepth(seldepth),
          m_prev(nullptr), m_excluded_move(MOVE_NULL), m_thread_id(thread_id),
          m_flags(NONE), m_static_eval(SCORE_NONE), m_move(MOVE_NULL)
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
        result.m_static_eval = SCORE_NONE;
        result.m_ply++;
        result.m_excluded_move = MOVE_NULL;
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


    void SearchData::exclude(Move move)
    {
        m_excluded_move = move;
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
        : m_histories(std::make_unique<Histories>()),
          m_nodes_searched(0),
          m_seldepth(0),
          m_data(*m_histories, id, m_nodes_searched, m_seldepth)
    {
    }


    Histories& SearchThread::histories()
    {
        return *m_histories;
    }


    SearchData& SearchThread::data()
    {
        return m_data;
    }



    bool timeout()
    {
        if (!thinking)
            return true;

        // Never timeout on ponder mode
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
        if (limits.time[turn] != 0)
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



    void start_search_threads()
    {
        thinking = true;
        nodes_searched = 0;
        threads.reserve(Parameters::n_threads);
        for (int i = 0; i < Parameters::n_threads; i++)
            threads.push_back(std::thread(thread_search, i));
    }



    void stop_search_threads()
    {
        thinking = false;
        for (auto& thread : threads)
            if (thread.joinable())
                thread.join();

        threads.clear();
    }



    void go_perft(Depth depth)
    {
        Position position = *base_position;
        int64_t n_nodes = perft<true>(position, depth);
        std::cout << "\nNodes searched: " << n_nodes << std::endl;
    }



    void thread_search(int id)
    {
        Position position = *base_position;
        auto thread_data = std::make_unique<SearchThread>(id);
        iter_deepening(position, thread_data->data());
    }



    void get_pv(Position& position, Depth depth, MoveList& pv)
    {
        Depth curr_depth = 0;
        TranspositionEntry* entry = nullptr;
        pv.clear();
        Move move;
        while (ttable.query(position.hash(), &entry) &&
               (move = entry->hash_move()) != MOVE_NULL &&
               position.board().legal(move) &&
               (curr_depth++) < depth &&
               !position.is_draw(false))
        {
            pv.push(move);
            position.make_move(move);
        }

        // Pv is populated, undo the moves
        for (auto move : pv)
            position.unmake_move();
    }



    Score aspiration_search(Position& position, Score init_score, Depth depth, Color color, SearchData& data)
    {
        constexpr Score starting_window = 25;
        Score l_window = starting_window;
        Score r_window = starting_window;

        // Initial search
        Score alpha = (depth <= 4) ? (-SCORE_INFINITE) : std::max(-SCORE_INFINITE, (init_score - l_window));
        Score beta  = (depth <= 4) ? ( SCORE_INFINITE) : std::min(+SCORE_INFINITE, (init_score + r_window));

        Score final_score = negamax<PV>(position, depth, alpha, beta, data);

        // Increase window in the failed side exponentially
        while ((final_score <= alpha || final_score >= beta) && !timeout())
        {
            bool upperbound = final_score <= alpha;
            Score bound = upperbound ? alpha : beta;
            if (upperbound)
                l_window *= 2;
            else
                r_window *= 2;

            // Output some information
            if (data.thread() == 0)
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

            // Repeat search
            final_score = negamax<PV>(position, depth, alpha, beta, data);
        }

        return final_score;
    }



    Score iter_deepening(Position& position, SearchData& data)
    {
        // Maximum PV lines
        int maxPV = std::min(position.generate_moves(MoveGenType::LEGAL).lenght(), Parameters::multiPV);

        Score score = -SCORE_INFINITE;
        MoveStack pv_stack(maxPV);
        Color color = turn_to_color(position.get_turn());
        std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();
        bool main_thread = data.thread() == 0;
        if (main_thread)
        {
            multiPV_roots.resize(maxPV);
            multiPV_scores.resize(maxPV);
            multiPV_lists.resize(maxPV);
        }

        Move bestmove = MOVE_NULL;
        Move pondermove = MOVE_NULL;

        // Iterative deepening
        position.set_init_ply();
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
                if (timeout())
                    break;

                // Store depth results
                if (main_thread)
                {
                    get_pv(position, iDepth, multiPV_lists[iPv]);
                    multiPV_scores[iPv] = curr_score;
                    multiPV_roots[iPv] = *multiPV_lists[iPv].begin();
                }
            }

            // Timeout?
            if (timeout())
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
                    std::cout << " seldepth " << data.seldepth() + iDepth;
                    std::cout << " multipv " << iPv + 1;
                    if (is_mate(multiPV_scores[iPv]))
                        std::cout << " score mate " << mate_in(multiPV_scores[iPv]);
                    else
                        std::cout << " score cp " << multiPV_scores[iPv];
                    std::cout << " nodes " << nodes_searched;
                    std::cout << " nps " << static_cast<int>(nodes_searched / elapsed);
                    std::cout << " hashfull " << ttable.hashfull();
                    std::cout << " time " << std::max(1, static_cast<int>(time_depth * 1000));
                    std::cout << " pv " << multiPV_lists[iPv];
                    std::cout << std::endl;
                }
            }

            // Update best and ponder moves
            auto best_pv = multiPV_lists[0];
            auto move_ptr = best_pv.begin();
            bestmove = *move_ptr;
            pondermove = (best_pv.lenght() > 1) ? *(move_ptr + 1) : MOVE_NULL;

            // Additional stopping conditions
            score = multiPV_scores[0];
            if (main_thread &&
                !Parameters::limits.ponder &&
                !Parameters::limits.infinite)
            {
                double remaining = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - std::chrono::steady_clock::now()).count() / 1e9;
                // Best move mate or do we expect not to have time for one more iteration?
                if (abs(score) > SCORE_MATE_FOUND ||
                    (remaining > 0 && remaining < time_depth * 1.5))
                {
                    thinking = false;
                    break;
                }
            }
        }

        // Output bestmove
        if (main_thread)
        {
            std::cout << "bestmove " << bestmove;
            if (pondermove != MOVE_NULL)
                std::cout << " ponder " << pondermove;
            std::cout << std::endl;
        }

        return score;
    }



    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data)
    {
        // Node data
        constexpr bool PvNode = ST == PV;
        const bool RootSearch = data.ply() == 1;
        const bool Reduced = data.reductions() > 0;
        const bool Extended = data.extensions() > 0;
        const bool HasExcludedMove = data.excluded_move() != MOVE_NULL;
        const bool IsCheck = position.is_check();
        const Turn Turn = position.get_turn();
        const Depth Ply = position.ply();

        // Timeout?
        if (depth > 1 && timeout())
            return SCORE_NONE;

        // Mate distance prunning: don't bother searching if we are deeper than the shortest mate up to this point
        if (!RootSearch)
        {
            alpha = std::max(alpha, static_cast<Score>(-SCORE_MATE + position.ply()));
            beta  = std::min(beta,  static_cast<Score>( SCORE_MATE - position.ply()));
            if (alpha >= beta)
                return alpha;
        }

        // Dive into quiescence at leaf nodes
        if (depth <= 0)
            return quiescence(position, 0, alpha, beta, data);

        // Early check for draw
        if (position.is_draw(!RootSearch))
            return SCORE_DRAW;

        // TT lookup
        Score alpha_init = alpha;
        Move tt_move = MOVE_NULL;
        Score tt_score = SCORE_NONE;
        Score tt_static_eval = SCORE_NONE;
        TranspositionEntry* entry = nullptr;
        bool tt_hit = ttable.query(position.hash(), &entry);
        if (tt_hit)
        {
            tt_score = entry->score();
            tt_move = entry->hash_move();
            tt_static_eval = entry->static_eval();

            // TT cutoff in non-PV nodes
            if (!PvNode && entry->depth() >= depth)
            {
                auto type = entry->type();
                if (type == EntryType::EXACT ||
                    type == EntryType::UPPER_BOUND && tt_score <= alpha ||
                    type == EntryType::LOWER_BOUND && tt_score >= beta)
                    return score_from_tt(tt_score, Ply);
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

        // Futility pruning
        if (depth < 5 && !position.is_check() &&
            !is_mate(alpha) && !is_mate(beta))
        {
            Score margin = 200 * depth;
            if (static_eval - margin >= beta || static_eval + margin <= alpha)
                return static_eval;
        }

        // Null move pruning
        if (!PvNode && !IsCheck && !HasExcludedMove &&
            static_eval >= beta &&
            data.last_move() != MOVE_NULL &&
            position.board().sliders())
        {
            int reduction = 3;
            Depth new_depth = std::max(0, depth - 1 - reduction);
            SearchData curr_data = data.next(MOVE_NULL) | REDUCED;
            position.make_null_move();
            nodes_searched++;
            data.nodes_searched()++;
            Score null = -negamax<NON_PV>(position, new_depth, -beta, -beta + 1, curr_data);
            position.unmake_null_move();
            if (null >= beta && !is_mate(null))
                return null;
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
        MoveOrder orderer = MoveOrder(position, depth, tt_move, data.histories(), data.last_move());
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            n_moves++;
            if (!move.is_capture() && !move.is_promotion())
                move_number++;

            // Skip excluded moves
            if (move == data.excluded_move())
                continue;

            // New search parameters
            Depth curr_depth = depth;
            SearchData curr_data = data.next(move);
            uint64_t move_nodes = data.nodes_searched();

            // In multiPV mode do not search previous PV root nodes
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
                std::chrono::steady_clock::now() > start_time + std::chrono::milliseconds(3000))
                std::cout << "info depth " << static_cast<int>(depth)
                          << " currmove " << move.to_uci()
                          << " currmovenumber " << n_moves << std::endl;

            // Singular extensions: when the stored TT value fails high, we carry a reduced search on the remaining moves
            // If all moves fail low then we extend the TT move
            if (!RootSearch && tt_hit &&
                depth > 8 &&
                !HasExcludedMove &&
                !IsCheck &&
                move == tt_move &&
                entry->depth() >= depth - 3 &&
                entry->type() == EntryType::LOWER_BOUND &&
                !is_mate(tt_score) &&
                data.extensions() < 3)
            {
                Value singularBeta = entry->score() - 2 * depth;
                Depth singularDepth = (depth - 1) / 2;

                // Search with the move excluded
                curr_data.exclude(move);
                Score score = negamax<NON_PV>(position, singularDepth, singularBeta - 1, singularBeta, curr_data);
                curr_data.exclude(MOVE_NULL);

                if (score < singularBeta)
                {
                    // TT move is singular, we are extending it
                    curr_data |= EXTENDED;
                    curr_depth++;
                }
                else if (singularBeta >= beta)
                {
                    // Multi-cut prunning: assuming our TT move fails high, at least one more move also fails high
                    // So we can probably safely prune the entire tree
                    return singularBeta;
                }
            }

            // Make the move
            Score score;
            PieceType piece = static_cast<PieceType>(position.board().get_piece_at(move.from()));
            position.make_move(move);
            nodes_searched++;
            data.nodes_searched()++;

            // Late move reductions
            bool do_full_search = true;
            if (depth > 4 &&
                move_number > 3 &&
                !move.is_capture() &&
                !move.is_promotion() &&
                !position.is_check() &&
                !IsCheck &&
                !(data.flags() & EXTENDED) &&
                data.thread() % 3 < 2)
            {
                int reduction = 3 + (move_number - 4) / 8;
                Depth new_depth = std::min(depth - 1, std::max(1, depth - 1 - reduction));

                // Reduced depth search
                curr_data |= REDUCED;
                score = -negamax<NON_PV>(position, new_depth, -alpha - 1, -alpha, curr_data);
                curr_data ^= REDUCED;

                // Only carry a full search if this reduced move fails high
                do_full_search = score >= alpha;

                // Add a bonus for this move on fail high
                if (do_full_search)
                    data.histories().add_bonus(move, Turn, piece, depth);
            }

            // Check extensions
            if (IsCheck && data.extensions() < 3)
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
                    if (score > alpha&& score < beta)
                    {
                        // But before add a bonus to the move
                        data.histories().add_bonus(move, Turn, piece, depth);
                        score = -negamax<PV>(position, curr_depth - 1, -beta, -alpha, curr_data);
                    }
                }
            }

            // Unmake the move
            position.unmake_move();

            // Timeout?
            if (depth > 2 && timeout())
                return SCORE_NONE;

            // New best move
            if (score > best_score)
            {
                best_score = score;
                best_move = move;
                alpha = std::max(alpha, score);
            }

            // Update low-ply history
            move_nodes = data.nodes_searched() - move_nodes;
            if (Ply < NUM_LOW_PLY)
                data.histories().update_low_ply(move, Ply, piece, move_nodes / 10000);

            // Prunning
            if (alpha >= beta)
            {
                if (!move.is_capture())
                    data.histories().fail_high(move, data.last_move(), Turn, depth, Ply, piece);
                break;
            }
        }

        // Check for game end
        if (n_moves == 0)
        {
            // Checkmate or stalemate?
            if (position.is_check())
                best_score = -SCORE_MATE + position.ply();
            else
                best_score = SCORE_DRAW;
        }

        // TT store (except on partial searches and at root in non-main threads)
        if (data.excluded_move() == MOVE_NULL && !(RootSearch && data.thread() != 0))
        {
            EntryType type = best_score >= beta                  ? EntryType::LOWER_BOUND
                           : (PvNode && best_score > alpha_init) ? EntryType::EXACT
                           :                                       EntryType::UPPER_BOUND;
            ttable.store(TranspositionEntry(position.hash(), depth, score_to_tt(best_score, position.ply()), best_move, type, static_eval), RootSearch);
        }

        return best_score;
    }



    Score quiescence(Position& position, Depth depth, Score alpha, Score beta, SearchData& data)
    {
        bool is_check = position.is_check();

        // Early check for draw
        if (position.is_draw(true))
            return SCORE_DRAW;

        // Only call evaluation on non-check positions
        Score best_score = -SCORE_INFINITE;
        if (!is_check)
        {
            // Update seldepth and compute node score
            data.seldepth() = std::max(data.seldepth(), depth);
            best_score = turn_to_color(position.get_turn()) * evaluation(position);

            // Alpha-beta prunning
            alpha = std::max(best_score, alpha);
            if (alpha >= beta)
                return alpha;
        }

        // Search
        Move move;
        int n_moves = 0;
        MoveOrder orderer = MoveOrder(position, 0, MOVE_NULL, data.histories(), MOVE_NULL, true);
        while ((move = orderer.next_move()) != MOVE_NULL)
        {
            n_moves++;

            // Only search captures with positive SEE
            if (!is_check && position.board().see(move) < 0)
                continue;

            position.make_move(move);
            nodes_searched++;
            data.nodes_searched()++;
            Score score = -quiescence(position, depth + 1, -beta, -alpha, data);
            position.unmake_move();

            // New best value
            best_score = std::max(best_score, score);
            alpha = std::max(alpha, best_score);

            // Prunning
            if (alpha >= beta)
                break;
        }

        // Checkmate?
        if (n_moves == 0 && is_check)
            return -SCORE_MATE + position.ply();

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
        if (result != move_list.lenght())
        {
            std::cout << result << " vs " << move_list.lenght() << std::endl;
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
