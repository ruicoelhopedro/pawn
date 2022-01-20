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
        int64_t nodes;
        int mate;
        int movetime;
        bool infinite;
        int64_t end_time;

        Limits()
            : searchmoves(0)
        {
            ponder = infinite = false;
            time[WHITE] = time[BLACK] = incr[WHITE] = incr[BLACK] = 0;
            mate = 0;
            depth = NUM_MAX_DEPTH;
            movestogo = movetime = INT32_MAX;
            end_time = INT64_MAX;
            nodes = INT64_MAX;
        }
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
        int m_thread_id;
        Score m_static_eval;
        Move m_move;
        Move* m_pv;
        Move* m_prev_pv;
        bool m_isPv;

    public:
        SearchData(Histories& histories, int thread_id, int64_t& nodes_searched, Depth& seldepth, Move* pv, Move* prev_pv);

        SearchData next(Move move) const;

        Histories& histories() const;
        int reductions() const;
        int extensions() const;
        int ply() const;
        Move excluded_move() const;
        int thread() const;
        Score& static_eval();
        Score static_eval() const;
        Move last_move() const;
        SearchFlags flags() const;
        const SearchData* previous(int distance = 1) const;
        int64_t& nodes_searched();
        Depth& seldepth();
        bool in_pv() const;
        Move* pv();
        Move pv_move();

        void exclude(Move move);

        void update_pv(Move best_move, Move* new_pv);
        void accept_pv();
        void clear_pv();

        SearchData& operator|=(SearchFlags flag);
        SearchData& operator^=(SearchFlags flag);
        SearchData operator|(SearchFlags flag) const;
        SearchData operator^(SearchFlags flag) const;
    };


    class SearchThread
    {
        std::unique_ptr<Histories> m_histories;
        std::unique_ptr<PvContainer> m_pv;
        int64_t m_nodes_searched;
        Depth m_seldepth;
        SearchData m_data;

    public:
        SearchThread(int id);

        Histories& histories();
        SearchData& data();
    };


    extern bool thinking;
    extern int64_t nodes_searched;
    extern Position* base_position;
    extern std::vector<std::thread> threads;
    extern std::chrono::steady_clock::time_point start_time;
    extern std::chrono::steady_clock::time_point end_time;


    namespace Parameters
    {
        extern Depth depth;
        extern int multiPV;
        extern int n_threads;
        extern Limits limits;
        extern bool ponder;
    }


    bool timeout();


    void start_search_threads();


    void stop_search_threads();


    void go_perft(Depth depth);


    void thread_search(int id);


    void update_time();


    void get_pv(Position& position, Depth depth, MoveList& pv);


    Score aspiration_search(Position& position, Score init_score, Depth depth, Color color, SearchData& data);


    Score iter_deepening(Position& position, SearchData& data);


    template<SearchType ST>
    Score negamax(Position& position, Depth depth, Score alpha, Score beta, SearchData& data);


    template<SearchType ST>
    Score quiescence(Position& position, Score alpha, Score beta, SearchData& data);


    bool legality_tests(Position& position, MoveList& move_list);


    template<bool OUTPUT, bool USE_ORDER = false, bool TT = false, bool LEGALITY = false>
    int64_t perft(Position& position, Depth depth)
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
            MoveOrder orderer = MoveOrder(position, depth, 0, MOVE_NULL, Histories(), MOVE_NULL);
            while ((move = orderer.next_move()) != MOVE_NULL)
            {
                if (LEGALITY && !legality_tests(position, move_list))
                    return 0;

                int64_t count = 1;
                if (depth > 1)
                {
                    position.make_move(move);
                    count = perft<false, USE_ORDER, TT>(position, depth - 1);
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
                    count = perft<false, USE_ORDER, TT>(position, depth - 1);
                    position.unmake_move();

                    n_nodes += count;

                    if (OUTPUT)
                        std::cout << move.to_uci() << ": " << count << std::endl;
                }
            }
            else
            {
                n_nodes = move_list.lenght();
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
