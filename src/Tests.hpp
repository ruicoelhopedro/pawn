#pragma once
#include "Hash.hpp"
#include "Search.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "UCI.hpp"
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


    template<bool USE_ORDER, bool LEGALITY, bool VALIDITY>
    int perft_techniques_tests()
    {
        // Store initial state
        bool Chess960 = UCI::Options::UCI_Chess960;

        auto tests = test_suite();

        int n_failed = 0;
        for (auto& test : tests)
        {
            // Check if this is a FRC position
            UCI::Options::UCI_Chess960 = (test.fen().find("(FRC)") != std::string::npos);

            Position pos(test.fen());
            auto hists = std::make_unique<Histories>();
            Depth depth = test.depth() - 1 - 2 * (LEGALITY || VALIDITY);
            auto result_base = Search::perft<false>(pos, depth, *hists);
            auto result_test = Search::template perft<false, USE_ORDER, LEGALITY, VALIDITY>(pos, depth, *hists);
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

        UCI::Options::UCI_Chess960 = Chess960;
        std::cout << "\nFailed/total tests: " << n_failed << "/" << tests.size() << std::endl;
        return n_failed;
    }


    int legality_tests();
}
