#pragma once
#include "Types.hpp"
#include "Move.hpp"
#include "Position.hpp"
#include "Transpositions.hpp"
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
    struct Limits
    {
        MoveList searchmoves;
        bool ponder;
        int time[NUM_COLORS];
        int incr[NUM_COLORS];
        int movestogo;
        int depth;
        int64_t nodes;
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
            nodes = INT64_MAX;
        }
    };


    class Time
    {
        bool m_ponder;
        bool m_managed;
        int m_movetime;
        std::chrono::steady_clock::time_point m_start;
        std::chrono::steady_clock::time_point m_end;

        int diff(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) const;

    public:
        Time(bool ponder = false);
        Time(int movetime_ms, bool ponder = false);

        void ponderhit();

        int elapsed() const;
        int remaining() const;
        bool time_management() const;
        bool pondering() const;
    };


    class MultiPVData
    {
    public:
        Move bestmove;
        Score score;
        MoveList pv;
    
        MultiPVData();
    };


    enum SearchType
    {
        ROOT,
        PV,
        NON_PV,
    };

    enum SearchFlags
    {
        NONE = 0,
        ROOT_SEARCH = 1,
        REDUCED = 2,
        EXTENDED = 4,
        DOUBLE_EXTENDED = 8,
    };

    inline SearchFlags  operator~  (SearchFlags  a) { return (SearchFlags)~(int)a; }
    inline SearchFlags& operator|= (SearchFlags& a, SearchFlags b) { return (SearchFlags&)((int&)a |= (int)b); }
    inline SearchFlags& operator^= (SearchFlags& a, SearchFlags b) { return (SearchFlags&)((int&)a ^= (int)b); }
    inline SearchFlags& operator&= (SearchFlags& a, SearchFlags b) { return (SearchFlags&)((int&)a &= (int)b); }
    inline SearchFlags  operator|  (SearchFlags  a, SearchFlags b) { return (SearchFlags)((int)a | (int)b); }
    inline SearchFlags  operator^  (SearchFlags  a, SearchFlags b) { return (SearchFlags)((int)a ^ (int)b); }
    inline SearchFlags  operator&  (SearchFlags  a, SearchFlags b) { return (SearchFlags)((int)a & (int)b); }


    class SearchData
    {
        int m_ply;
        int64_t& m_nodes_searched;
        Depth& m_seldepth;
        SearchFlags m_flags;
        int m_reductions;
        int m_extensions;
        Histories& m_histories;
        const SearchData* m_prev;
        Move m_excluded_move;
        Thread& m_thread;
        Score m_static_eval;
        Move m_move;
        Move* m_pv;
        MoveList& m_prev_pv;
        bool m_isPv;

    public:
        SearchData(Thread& thread);

        SearchData next(Move move) const;

        int ply() const;
        int reductions() const;
        int extensions() const;
        bool in_pv() const;
        Move pv_move();
        Move last_move() const;
        Move excluded_move() const;
        Move* pv();
        Score static_eval() const;
        Score& static_eval();
        Depth& seldepth();
        Thread& thread() const;
        int64_t& nodes_searched();
        Histories& histories() const;
        SearchFlags flags() const;
        const Time& time() const;

        const SearchData* previous(int distance = 1) const;

        void exclude(Move move);

        void update_pv(Move best_move, Move* new_pv);
        void accept_pv();
        void clear_pv();

        bool timeout() const;

        SearchData& operator|=(SearchFlags flag);
        SearchData& operator^=(SearchFlags flag);
        SearchData operator|(SearchFlags flag) const;
        SearchData operator^(SearchFlags flag) const;
    };


    namespace Parameters
    {
        extern int multiPV;
        extern bool ponder;
    }


    void go_perft(Depth depth);


    Time update_time(const Position& position, Limits limits);


    void get_pv(SearchData& data, MoveList& pv);


    Score aspiration_search(Position& position, Score init_score, Depth depth, SearchData& data);


    Score iter_deepening(Position& position, SearchData& data);


    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data);


    template<SearchType ST>
    Score quiescence(Position& position, Score alpha, Score beta, SearchData& data);


    bool legality_tests(Position& position, MoveList& move_list);


    template<bool OUTPUT, bool USE_ORDER = false, bool TT = false, bool LEGALITY = false, bool VALIDITY = false>
    int64_t perft(Position& position, Depth depth)
    {
        // TT lookup
        PerftEntry* entry = nullptr;
        if (TT && perft_table.query(position.hash(), &entry) && entry->depth() == depth)
            return entry->n_nodes();

        // Move generation
        int64_t n_nodes = 0;
        auto move_list = position.generate_moves(MoveGenType::LEGAL);

        if (VALIDITY && !position.board().is_valid())
            return 0;

        // Move counting
        if (USE_ORDER)
        {
            // Use move orderer (slower but the actual method used during search)
            Move move;
            Histories hists;
            HistoryContext hc(hists, position.board(), 0, depth, MOVE_NULL);
            MoveOrder orderer = MoveOrder(position, MOVE_NULL, hc);
            while ((move = orderer.next_move()) != MOVE_NULL)
            {
                if (LEGALITY && !legality_tests(position, move_list))
                    return 0;

                int64_t count = 1;
                if (depth > 1)
                {
                    position.make_move(move);
                    count = perft<false, USE_ORDER, TT, LEGALITY, VALIDITY>(position, depth - 1);
                    position.unmake_move();
                }
                n_nodes += count;

                if (OUTPUT)
                    std::cout << move.to_uci() << ": " << count << std::endl;
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

                    position.make_move(move);
                    count = perft<false, USE_ORDER, TT, LEGALITY, VALIDITY>(position, depth - 1);
                    position.unmake_move();

                    n_nodes += count;

                    if (OUTPUT)
                        std::cout << move.to_uci() << ": " << count << std::endl;
                }
            }
            else
            {
                n_nodes = move_list.length();
                if (OUTPUT)
                    for (auto move : move_list)
                        std::cout << move.to_uci() << ": " << 1 << std::endl;
            }
        }

        // TT storing
        if (TT)
            perft_table.store(PerftEntry(position.hash(), depth, n_nodes));

        return n_nodes;
    }
}
