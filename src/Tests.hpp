#pragma once
#include "Hash.hpp"
#include "Search.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include <fstream>
#include <string>


namespace Tests
{
    class PerftTest
    {
        std::string m_fen;
        Depth m_depth;
        int64_t m_result;

    public:
        PerftTest(std::string fen, Depth depth, int64_t result)
            : m_fen(fen), m_depth(depth), m_result(result)
        {}

        std::string fen() const { return m_fen; }
        Depth depth() const { return m_depth; }
        int64_t result() const { return m_result; }
    };


    std::vector<PerftTest> test_suite();
    std::vector<std::string> bench_suite();


    int perft_tests();
    

    void bench(Search::Limits limits, int threads, int hash);


    template<bool USE_ORDER, bool TT, bool LEGALITY, bool VALIDITY>
    int perft_techniques_tests()
    {
        auto tests = test_suite();

        // Allocate TT
        if (TT)
            perft_table.resize(16);

        int n_failed = 0;
        for (auto& test : tests)
        {
            Histories hists;
            Position pos(test.fen());
            auto result_base = Search::perft<false>(pos, test.depth() - 1, hists);
            auto result_test = Search::template perft<false, USE_ORDER, TT, LEGALITY, VALIDITY>(pos, test.depth() - 1, hists);
            if (result_base == result_test)
            {
                std::cout << "[ OK ] " << test.fen() << " (" << result_test << ")" << std::endl;
            }
            else
            {
                std::cout << "[FAIL] " << test.fen() << " (base " << result_base << ", test " << result_test << ")" << std::endl;
                n_failed++;
            }
        }

        // Deallocate TT
        if (TT)
            perft_table.resize(0);

        std::cout << "\nFailed/total tests: " << n_failed << "/" << tests.size() << std::endl;
        return n_failed;
    }


    int legality_tests();
}
