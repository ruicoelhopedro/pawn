#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Hash.hpp"
#include "MoveOrder.hpp"
#include <atomic>
#include <memory>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>

class Thread;

namespace Search
{
    constexpr int PV_LENGTH = NUM_MAX_MOVES;
    constexpr int TOTAL_PV_LENGTH = (PV_LENGTH * PV_LENGTH + PV_LENGTH) / 2;

    struct PvContainer
    {
        Move pv[TOTAL_PV_LENGTH];
        Move prev_pv[PV_LENGTH];
    };

    struct Limits
    {
        std::vector<Move> searchmoves;
        bool ponder;
        int time[NUM_COLORS];
        int incr[NUM_COLORS];
        int movestogo;
        int depth;
        uint64_t nodes;
        int mate;
        int movetime;
        bool infinite;

        Limits()
        {
            ponder = infinite = false;
            time[WHITE] = time[BLACK] = -1;
            incr[WHITE] = incr[BLACK] = 0;
            mate = 0;
            depth = NUM_MAX_DEPTH;
            movestogo = movetime = -1;
            nodes = UINT64_MAX;
        }
    };


    class Timer
    {
        std::chrono::steady_clock::time_point m_start;


    public:
        Timer();

        double elapsed() const;
        std::chrono::steady_clock::time_point begin() const;

        static double diff(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b);
    };


    class SearchTime
    {
        Timer m_timer;
        bool m_managed;
        double m_optimum;
        uint64_t m_movetime_ms;
        std::atomic_bool m_pondering;
        std::atomic<std::chrono::steady_clock::time_point> m_end_time;

    public:
        SearchTime() noexcept;

        void init(const Timer& timer, bool ponder);
        void init(const Timer& timer, uint64_t movetime_ms, bool ponder);
        void init(const Timer& timer, uint64_t movetime_ms, uint64_t optimum_ms, bool ponder);

        void ponderhit();

        bool pondering() const;

        double elapsed() const;

        double optimum() const;

        double remaining() const;

        bool time_management() const;
    };


    enum class BoundType
    {
        LOWER_BOUND,
        UPPER_BOUND,
        EXACT,
        NO_BOUND
    };

    class MultiPVData
    {
    public:
        Depth depth;
        Depth seldepth;
        Score search_score;
        Move pv[NUM_MAX_DEPTH];
        BoundType search_bound;
        Score tb_score;
    
        MultiPVData();

        Score score() const;
        BoundType bound() const;
        void write_pv(const Board& board, int index, uint64_t nodes, uint64_t tb_hits, double elapsed) const;
    };


    enum SearchType
    {
        ROOT,
        PV,
        NON_PV,
    };


    class SearchData
    {
        int m_ply;
        const SearchData* m_prev;
        Thread& m_thread;
        Move m_move;
        Move* m_pv;
        Move* m_prev_pv;
        bool m_isPv;

    public:
        SearchData(Thread& thread);

        SearchData next(Move move) const;

        Depth& seldepth;
        Score static_eval;
        Move excluded_move;
        Histories& histories;

        int ply() const;
        bool in_pv() const;
        Move pv_move();
        Move last_move() const;
        Move* pv();
        Move* prev_pv();
        Thread& thread() const;
        uint64_t nodes_searched() const;

        const SearchData* previous(int distance = 1) const;

        void update_pv(Move best_move, Move* new_pv);
        void accept_pv();
        void clear_pv();
    };


    void copy_pv(Move* src, Move* dst);


    Score aspiration_search(Position& position, MultiPVData& pv, Depth depth, SearchData& data);


    Score iter_deepening(Position& position, SearchData& data);


    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data, bool cut_node);


    template<SearchType ST>
    Score quiescence(Position& position, Score alpha, Score beta, SearchData& data);


    bool legality_tests(Position& position, MoveList& move_list);


    template<bool OUTPUT, bool USE_ORDER = false, bool TT = false, bool LEGALITY = false, bool VALIDITY = false>
    int64_t perft(Position& position, Depth depth, Histories& hists)
    {
        // TT lookup
        PerftEntry* entry = nullptr;
        if (TT && perft_table.query(position.hash(), &entry) && entry->depth() == depth)
            return entry->n_nodes();

        // Move generation
        int64_t n_nodes = 0;
        auto move_list = position.generate_moves(MoveGenType::LEGAL);

        // Move counting
        if (USE_ORDER)
        {
            // Use move orderer (slower but the actual method used during search)
            Move move;
            CurrentHistory history = hists.get(position);
            MoveOrder orderer = MoveOrder(position, depth, MOVE_NULL, history);
            while ((move = orderer.next_move()) != MOVE_NULL)
            {
                if (LEGALITY && !legality_tests(position, move_list))
                    return 0;

                if (VALIDITY && !position.board().is_valid())
                    return 0;

                int64_t count = 1;
                if (depth > 1)
                {
                    position.make_move(move);
                    count = perft<false, USE_ORDER, TT, LEGALITY, VALIDITY>(position, depth - 1, hists);
                    position.unmake_move();
                }
                n_nodes += count;

                if (OUTPUT)
                    std::cout << position.board().to_uci(move) << ": " << count << std::endl;
            }
        }
        else
        {
            // Use moves as they come from the generator (faster)
            int64_t count;
            if (depth > 1)
            {
                for (auto move : move_list)
                {
                    if (LEGALITY && !legality_tests(position, move_list))
                        return 0;

                    if (VALIDITY && !position.board().is_valid())
                        return 0;

                    position.make_move(move);
                    count = perft<false, USE_ORDER, TT, LEGALITY, VALIDITY>(position, depth - 1, hists);
                    position.unmake_move();

                    n_nodes += count;

                    if (OUTPUT)
                        std::cout << position.board().to_uci(move) << ": " << count << std::endl;
                }
            }
            else
            {
                n_nodes = move_list.length();
                if (OUTPUT)
                    for (auto move : move_list)
                        std::cout << position.board().to_uci(move) << ": " << 1 << std::endl;
            }
        }

        // TT storing
        if (TT)
            perft_table.store(position.hash(), depth, n_nodes);

        return n_nodes;
    }

    constexpr int ilog2(int v) { return Bitboard(v).bitscan_reverse(); }
}
