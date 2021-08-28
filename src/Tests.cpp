#include "Move.hpp"
#include "Position.hpp"
#include "Search.hpp"
#include "Tests.hpp"
#include "Types.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace Tests
{

    std::vector<PerftTest> test_suite()
    {
        std::vector<PerftTest> tests;

        tests.push_back(PerftTest("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324));
        tests.push_back(PerftTest("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1134888));
        tests.push_back(PerftTest("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1015133));
        tests.push_back(PerftTest("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467));
        tests.push_back(PerftTest("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661072));
        tests.push_back(PerftTest("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803711));
        tests.push_back(PerftTest("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206));
        tests.push_back(PerftTest("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476));
        tests.push_back(PerftTest("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001));
        tests.push_back(PerftTest("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1004658));
        tests.push_back(PerftTest("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217342));
        tests.push_back(PerftTest("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92683));
        tests.push_back(PerftTest("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2217));
        tests.push_back(PerftTest("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567584));
        tests.push_back(PerftTest("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23527));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 6, 764643));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 846648));
        tests.push_back(PerftTest("4k2r/8/8/8/8/8/8/4K3 w k - 0 1", 6, 899442));
        tests.push_back(PerftTest("r3k3/8/8/8/8/8/8/4K3 w q - 0 1", 6, 1001523));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", 6, 2788982));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", 6, 3517770));
        tests.push_back(PerftTest("8/8/8/8/8/8/6k1/4K2R w K - 0 1", 6, 185867));
        tests.push_back(PerftTest("8/8/8/8/8/8/1k6/R3K3 w Q - 0 1", 6, 413018));
        tests.push_back(PerftTest("4k2r/6K1/8/8/8/8/8/8 w k - 0 1", 6, 179869));
        tests.push_back(PerftTest("r3k3/1K6/8/8/8/8/8/8 w q - 0 1", 6, 367724));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 6, 179862938));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/1R2K2R w Kkq - 0 1", 6, 195629489));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/2R1K2R w Kkq - 0 1", 6, 184411439));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K1R1 w Qkq - 0 1", 6, 189224276));
        tests.push_back(PerftTest("1r2k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1", 6, 198328929));
        tests.push_back(PerftTest("2r1k2r/8/8/8/8/8/8/R3K2R w KQk - 0 1", 6, 185959088));
        tests.push_back(PerftTest("r3k1r1/8/8/8/8/8/8/R3K2R w KQq - 0 1", 6, 190755813));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/4K2R b K - 0 1", 6, 899442));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K3 b Q - 0 1", 6, 1001523));
        tests.push_back(PerftTest("4k2r/8/8/8/8/8/8/4K3 b k - 0 1", 6, 764643));
        tests.push_back(PerftTest("r3k3/8/8/8/8/8/8/4K3 b q - 0 1", 6, 846648));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1", 6, 3517770));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", 6, 2788982));
        tests.push_back(PerftTest("8/8/8/8/8/8/6k1/4K2R b K - 0 1", 6, 179869));
        tests.push_back(PerftTest("8/8/8/8/8/8/1k6/R3K3 b Q - 0 1", 6, 367724));
        tests.push_back(PerftTest("4k2r/6K1/8/8/8/8/8/8 b k - 0 1", 6, 185867));
        tests.push_back(PerftTest("r3k3/1K6/8/8/8/8/8/8 b q - 0 1", 6, 413018));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 6, 179862938));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/1R2K2R b Kkq - 0 1", 6, 198328929));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/2R1K2R b Kkq - 0 1", 6, 185959088));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K1R1 b Qkq - 0 1", 6, 190755813));
        tests.push_back(PerftTest("1r2k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1", 6, 195629489));
        tests.push_back(PerftTest("2r1k2r/8/8/8/8/8/8/R3K2R b KQk - 0 1", 6, 184411439));
        tests.push_back(PerftTest("r3k1r1/8/8/8/8/8/8/R3K2R b KQq - 0 1", 6, 189224276));
        tests.push_back(PerftTest("8/1n4N1/2k5/8/8/5K2/1N4n1/8 w - - 0 1", 6, 8107539));
        tests.push_back(PerftTest("8/1k6/8/5N2/8/4n3/8/2K5 w - - 0 1", 6, 2594412));
        tests.push_back(PerftTest("8/8/4k3/3Nn3/3nN3/4K3/8/8 w - - 0 1", 6, 19870403));
        tests.push_back(PerftTest("K7/8/2n5/1n6/8/8/8/k6N w - - 0 1", 6, 588695));
        tests.push_back(PerftTest("k7/8/2N5/1N6/8/8/8/K6n w - - 0 1", 6, 688780));
        tests.push_back(PerftTest("8/1n4N1/2k5/8/8/5K2/1N4n1/8 b - - 0 1", 6, 8503277));
        tests.push_back(PerftTest("8/1k6/8/5N2/8/4n3/8/2K5 b - - 0 1", 6, 3147566));
        tests.push_back(PerftTest("8/8/3K4/3Nn3/3nN3/4k3/8/8 b - - 0 1", 6, 4405103));
        tests.push_back(PerftTest("K7/8/2n5/1n6/8/8/8/k6N b - - 0 1", 6, 688780));
        tests.push_back(PerftTest("k7/8/2N5/1N6/8/8/8/K6n b - - 0 1", 6, 588695));
        tests.push_back(PerftTest("B6b/8/8/8/2K5/4k3/8/b6B w - - 0 1", 6, 22823890));
        tests.push_back(PerftTest("8/8/1B6/7b/7k/8/2B1b3/7K w - - 0 1", 6, 28861171));
        tests.push_back(PerftTest("k7/B7/1B6/1B6/8/8/8/K6b w - - 0 1", 6, 7881673));
        tests.push_back(PerftTest("K7/b7/1b6/1b6/8/8/8/k6B w - - 0 1", 6, 7382896));
        tests.push_back(PerftTest("B6b/8/8/8/2K5/5k2/8/b6B b - - 0 1", 6, 9250746));
        tests.push_back(PerftTest("8/8/1B6/7b/7k/8/2B1b3/7K b - - 0 1", 6, 29027891));
        tests.push_back(PerftTest("k7/B7/1B6/1B6/8/8/8/K6b b - - 0 1", 6, 7382896));
        tests.push_back(PerftTest("K7/b7/1b6/1b6/8/8/8/k6B b - - 0 1", 6, 7881673));
        tests.push_back(PerftTest("7k/RR6/8/8/8/8/rr6/7K w - - 0 1", 6, 44956585));
        tests.push_back(PerftTest("R6r/8/8/2K5/5k2/8/8/r6R w - - 0 1", 6, 525169084));
        tests.push_back(PerftTest("7k/RR6/8/8/8/8/rr6/7K b - - 0 1", 6, 44956585));
        tests.push_back(PerftTest("R6r/8/8/2K5/5k2/8/8/r6R b - - 0 1", 6, 524966748));
        tests.push_back(PerftTest("6kq/8/8/8/8/8/8/7K w - - 0 1", 6, 391507));
        tests.push_back(PerftTest("6KQ/8/8/8/8/8/8/7k b - - 0 1", 6, 391507));
        tests.push_back(PerftTest("K7/8/8/3Q4/4q3/8/8/7k w - - 0 1", 6, 3370175));
        tests.push_back(PerftTest("6qk/8/8/8/8/8/8/7K b - - 0 1", 6, 419369));
        tests.push_back(PerftTest("6KQ/8/8/8/8/8/8/7k b - - 0 1", 6, 391507));
        tests.push_back(PerftTest("K7/8/8/3Q4/4q3/8/8/7k b - - 0 1", 6, 3370175));
        tests.push_back(PerftTest("8/8/8/8/8/K7/P7/k7 w - - 0 1", 6, 6249));
        tests.push_back(PerftTest("8/8/8/8/8/7K/7P/7k w - - 0 1", 6, 6249));
        tests.push_back(PerftTest("K7/p7/k7/8/8/8/8/8 w - - 0 1", 6, 2343));
        tests.push_back(PerftTest("7K/7p/7k/8/8/8/8/8 w - - 0 1", 6, 2343));
        tests.push_back(PerftTest("8/2k1p3/3pP3/3P2K1/8/8/8/8 w - - 0 1", 6, 34834));
        tests.push_back(PerftTest("8/8/8/8/8/K7/P7/k7 b - - 0 1", 6, 2343));
        tests.push_back(PerftTest("8/8/8/8/8/7K/7P/7k b - - 0 1", 6, 2343));
        tests.push_back(PerftTest("K7/p7/k7/8/8/8/8/8 b - - 0 1", 6, 6249));
        tests.push_back(PerftTest("7K/7p/7k/8/8/8/8/8 b - - 0 1", 6, 6249));
        tests.push_back(PerftTest("8/2k1p3/3pP3/3P2K1/8/8/8/8 b - - 0 1", 6, 34822));
        tests.push_back(PerftTest("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", 6, 11848));
        tests.push_back(PerftTest("4k3/4p3/4K3/8/8/8/8/8 b - - 0 1", 6, 11848));
        tests.push_back(PerftTest("8/8/7k/7p/7P/7K/8/8 w - - 0 1", 6, 10724));
        tests.push_back(PerftTest("8/8/k7/p7/P7/K7/8/8 w - - 0 1", 6, 10724));
        tests.push_back(PerftTest("8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1", 6, 53138));
        tests.push_back(PerftTest("8/3k4/3p4/8/3P4/3K4/8/8 w - - 0 1", 6, 157093));
        tests.push_back(PerftTest("8/8/3k4/3p4/8/3P4/3K4/8 w - - 0 1", 6, 158065));
        tests.push_back(PerftTest("k7/8/3p4/8/3P4/8/8/7K w - - 0 1", 6, 20960));
        tests.push_back(PerftTest("8/8/7k/7p/7P/7K/8/8 b - - 0 1", 6, 10724));
        tests.push_back(PerftTest("8/8/k7/p7/P7/K7/8/8 b - - 0 1", 6, 10724));
        tests.push_back(PerftTest("8/8/3k4/3p4/3P4/3K4/8/8 b - - 0 1", 6, 53138));
        tests.push_back(PerftTest("8/3k4/3p4/8/3P4/3K4/8/8 b - - 0 1", 6, 158065));
        tests.push_back(PerftTest("8/8/3k4/3p4/8/3P4/3K4/8 b - - 0 1", 6, 157093));
        tests.push_back(PerftTest("k7/8/3p4/8/3P4/8/8/7K b - - 0 1", 6, 21104));
        tests.push_back(PerftTest("7k/3p4/8/8/3P4/8/8/K7 w - - 0 1", 6, 32191));
        tests.push_back(PerftTest("7k/8/8/3p4/8/8/3P4/K7 w - - 0 1", 6, 30980));
        tests.push_back(PerftTest("k7/8/8/7p/6P1/8/8/K7 w - - 0 1", 6, 41874));
        tests.push_back(PerftTest("k7/8/7p/8/8/6P1/8/K7 w - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/8/8/6p1/7P/8/8/K7 w - - 0 1", 6, 41874));
        tests.push_back(PerftTest("k7/8/6p1/8/8/7P/8/K7 w - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/8/8/3p4/4p3/8/8/7K w - - 0 1", 6, 22886));
        tests.push_back(PerftTest("k7/8/3p4/8/8/4P3/8/7K w - - 0 1", 6, 28662));
        tests.push_back(PerftTest("7k/3p4/8/8/3P4/8/8/K7 b - - 0 1", 6, 32167));
        tests.push_back(PerftTest("7k/8/8/3p4/8/8/3P4/K7 b - - 0 1", 6, 30749));
        tests.push_back(PerftTest("k7/8/8/7p/6P1/8/8/K7 b - - 0 1", 6, 41874));
        tests.push_back(PerftTest("k7/8/7p/8/8/6P1/8/K7 b - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/8/8/6p1/7P/8/8/K7 b - - 0 1", 6, 41874));
        tests.push_back(PerftTest("k7/8/6p1/8/8/7P/8/K7 b - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/8/8/3p4/4p3/8/8/7K b - - 0 1", 6, 22579));
        tests.push_back(PerftTest("k7/8/3p4/8/8/4P3/8/7K b - - 0 1", 6, 28662));
        tests.push_back(PerftTest("7k/8/8/p7/1P6/8/8/7K w - - 0 1", 6, 41874));
        tests.push_back(PerftTest("7k/8/p7/8/8/1P6/8/7K w - - 0 1", 6, 29679));
        tests.push_back(PerftTest("7k/8/8/1p6/P7/8/8/7K w - - 0 1", 6, 41874));
        tests.push_back(PerftTest("7k/8/1p6/8/8/P7/8/7K w - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/7p/8/8/8/8/6P1/K7 w - - 0 1", 6, 55338));
        tests.push_back(PerftTest("k7/6p1/8/8/8/8/7P/K7 w - - 0 1", 6, 55338));
        tests.push_back(PerftTest("3k4/3pp3/8/8/8/8/3PP3/3K4 w - - 0 1", 6, 199002));
        tests.push_back(PerftTest("7k/8/8/p7/1P6/8/8/7K b - - 0 1", 6, 41874));
        tests.push_back(PerftTest("7k/8/p7/8/8/1P6/8/7K b - - 0 1", 6, 29679));
        tests.push_back(PerftTest("7k/8/8/1p6/P7/8/8/7K b - - 0 1", 6, 41874));
        tests.push_back(PerftTest("7k/8/1p6/8/8/P7/8/7K b - - 0 1", 6, 29679));
        tests.push_back(PerftTest("k7/7p/8/8/8/8/6P1/K7 b - - 0 1", 6, 55338));
        tests.push_back(PerftTest("k7/6p1/8/8/8/8/7P/K7 b - - 0 1", 6, 55338));
        tests.push_back(PerftTest("3k4/3pp3/8/8/8/8/3PP3/3K4 b - - 0 1", 6, 199002));
        tests.push_back(PerftTest("8/Pk6/8/8/8/8/6Kp/8 w - - 0 1", 6, 1030499));
        tests.push_back(PerftTest("n1n5/1Pk5/8/8/8/8/5Kp1/5N1N w - - 0 1", 6, 37665329));
        tests.push_back(PerftTest("8/PPPk4/8/8/8/8/4Kppp/8 w - - 0 1", 6, 28859283));
        tests.push_back(PerftTest("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1", 6, 71179139));
        tests.push_back(PerftTest("8/Pk6/8/8/8/8/6Kp/8 b - - 0 1", 6, 1030499));
        tests.push_back(PerftTest("n1n5/1Pk5/8/8/8/8/5Kp1/5N1N b - - 0 1", 6, 37665329));
        tests.push_back(PerftTest("8/PPPk4/8/8/8/8/4Kppp/8 b - - 0 1", 6, 28859283));
        tests.push_back(PerftTest("n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 6, 71179139));
        tests.push_back(PerftTest("r2q1b1r/ppp1ppp1/n1k2n1p/3p4/P4P2/2NP3P/1PPKP1BP/R1BQ3R b - f3 0 8", 6, 542729390));
        tests.push_back(PerftTest("rnbqkbnr/Pp1ppppp/8/8/2P5/4B3/PP2PPPP/RN1QKBNR b KQkq - 0 4", 6, 632428291));
        tests.push_back(PerftTest("3r1r2/2q2p1k/p3b3/1p6/4Pp1P/P1Nn2P1/1PP5/R4R1K b - h3 0 4", 6, 2719206243));

        return tests;
    }


    int perft_tests()
    {
        auto tests = test_suite();

        int n_failed = 0;
        for (auto& test : tests)
        {
            Position pos(test.fen());
            auto result = Search::perft<false>(pos, test.depth());
            if (result == test.result())
            {
                std::cout << "[ OK ] " << test.fen() << " (" << result << ")" << std::endl;
            }
            else
            {
                std::cout << "[FAIL] " << test.fen() << " (ref " << test.result() << ", new " << result << ")" << std::endl;
                n_failed++;
            }
        }

        std::cout << "\nFailed/total tests: " << n_failed << "/" << tests.size() << std::endl;
        return n_failed;
    }


    int legality_tests()
    {
        auto tests = test_suite();

        int n_failed = 0;
        for (auto& test : tests)
        {
            Position pos(test.fen());

            auto moves = pos.generate_moves(MoveGenType::LEGAL);
            int reference = moves.lenght();

            int result = 0;
            for (uint16_t number = 0; number < UINT16_MAX; number++)
                if (pos.board().legal(Move::from_int(number)))
                    result++;

            if (result == reference)
            {
                std::cout << "[ OK ] " << test.fen() << " (" << result << ")" << std::endl;
            }
            else
            {
                std::cout << "[FAIL] " << test.fen() << " (ref " << reference << ", new " << result << ")" << std::endl;
                n_failed++;
            }
        }

        std::cout << "\nFailed/total tests: " << n_failed << "/" << tests.size() << std::endl;
        return n_failed;
    }
}
