#include "Move.hpp"
#include "Position.hpp"
#include "Search.hpp"
#include "Tests.hpp"
#include "Thread.hpp"
#include "Types.hpp"
#include "UCI.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace Tests
{

    std::vector<PerftTest> test_suite()
    {
        std::vector<PerftTest> tests;

        tests.push_back(PerftTest("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324));
        tests.push_back(PerftTest("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1440467));
        tests.push_back(PerftTest("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1274206));
        tests.push_back(PerftTest("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1720476));
        tests.push_back(PerftTest("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3821001));
        tests.push_back(PerftTest("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1004658));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 6, 764643));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 846648));
        tests.push_back(PerftTest("4k2r/8/8/8/8/8/8/4K3 w k - 0 1", 6, 899442));
        tests.push_back(PerftTest("r3k3/8/8/8/8/8/8/4K3 w q - 0 1", 6, 1001523));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", 6, 2788982));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/4K3 w kq - 0 1", 6, 3517770));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", 6, 179862938));
        tests.push_back(PerftTest("4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1", 6, 3517770));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", 6, 2788982));
        tests.push_back(PerftTest("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", 6, 179862938));
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
        tests.push_back(PerftTest("8/8/3k4/3p4/3P4/3K4/8/8 b - - 0 1", 6, 53138));
        tests.push_back(PerftTest("8/3k4/3p4/8/3P4/3K4/8/8 b - - 0 1", 6, 158065));
        tests.push_back(PerftTest("8/8/3k4/3p4/8/3P4/3K4/8 b - - 0 1", 6, 157093));
        tests.push_back(PerftTest("r2q1b1r/ppp1ppp1/n1k2n1p/3p4/P4P2/2NP3P/1PPKP1BP/R1BQ3R b - f3 0 8", 6, 542729390));
        tests.push_back(PerftTest("rnbqkbnr/Pp1ppppp/8/8/2P5/4B3/PP2PPPP/RN1QKBNR b KQkq - 0 4", 6, 632428291));
        tests.push_back(PerftTest("3r1r2/2q2p1k/p3b3/1p6/4Pp1P/P1Nn2P1/1PP5/R4R1K b - h3 0 4", 6, 2719206243));
        tests.push_back(PerftTest("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9 (FRC)", 6, 227689589));
        tests.push_back(PerftTest("2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9 (FRC)", 6, 590751109));
        tests.push_back(PerftTest("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9 (FRC)", 6, 177654692));
        tests.push_back(PerftTest("qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9 (FRC)", 6, 274103539));
        tests.push_back(PerftTest("1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9 (FRC)", 6, 1250970898));
        tests.push_back(PerftTest("1qnrkbbr/1pppppp1/p1n4p/8/P7/1P1N1P2/2PPP1PP/QN1RKBBR w HDhd - 0 9 (FRC)", 6, 783201510));
        tests.push_back(PerftTest("qn1rkrbb/pp1p1ppp/2p1p3/3n4/4P2P/2NP4/PPP2PP1/Q1NRKRBB w FDfd - 1 9 (FRC)", 6, 233468620));
        tests.push_back(PerftTest("bb1qnrkr/pp1p1pp1/1np1p3/4N2p/8/1P4P1/P1PPPP1P/BBNQ1RKR w HFhf - 0 9 (FRC)", 6, 776836316));
        tests.push_back(PerftTest("bnqbnr1r/p1p1ppkp/3p4/1p4p1/P7/3NP2P/1PPP1PP1/BNQB1RKR w HF - 0 9 (FRC)", 6, 809194268));
        tests.push_back(PerftTest("bnqnrbkr/1pp2pp1/p7/3pP2p/4P1P1/8/PPPP3P/BNQNRBKR w HEhe d6 0 9 (FRC)", 6, 1008880643));
        tests.push_back(PerftTest("b1qnrrkb/ppp1pp1p/n2p1Pp1/8/8/P7/1PPPP1PP/BNQNRKRB w GE - 0 9 (FRC)", 6, 193594729));

        return tests;
    }


    std::vector<std::string> bench_suite()
    {
        std::vector<std::string> bench;

        bench.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        bench.push_back("r2q1b1r/ppp1ppp1/n1k2n1p/3p4/P4P2/2NP3P/1PPKP1BP/R1BQ3R b - f3 0 8");
        bench.push_back("rnbqkbnr/Pp1ppppp/8/8/2P5/4B3/PP2PPPP/RN1QKBNR b KQkq - 0 4");
        bench.push_back("3r1r2/2q2p1k/p3b3/1p6/4Pp1P/P1Nn2P1/1PP5/R4R1K b - h3 0 4");
        bench.push_back("8/Pk6/8/8/8/8/6Kp/8 w - - 0 1");
        bench.push_back("7k/3p4/8/8/3P4/8/8/K7 b - - 0 1");
        bench.push_back("k7/B7/1B6/1B6/8/8/8/K6b w - - 0 1");
        bench.push_back("5Q2/8/8/8/3b4/5Q2/Q7/K1k5 w - - 0 1");
        bench.push_back("8/8/5k2/8/8/8/8/KN4Q1 w - - 0 1");
        bench.push_back("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1");
        bench.push_back("3r1k2/4npp1/1ppr3p/p6P/P2PPPP1/1NR5/5K2/2R5 w - - 0 1");
        bench.push_back("2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - - 0 1");
        bench.push_back("rnbqkb1r/p3pppp/1p6/2ppP3/3N4/2P5/PPP1QPPP/R1B1KB1R w KQkq - 0 1");
        bench.push_back("r1b2rk1/2q1b1pp/p2ppn2/1p6/3QP3/1BN1B3/PPP3PP/R4RK1 w - - 0 1");
        bench.push_back("2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1");
        bench.push_back("1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/R2Q2K1 w - - 0 1");
        bench.push_back("4b3/p3kp2/6p1/3pP2p/2pP1P2/4K1P1/P3N2P/8 w - - 0 1");
        bench.push_back("2kr1bnr/pbpq4/2n1pp2/3p3p/3P1P1B/2N2N1Q/PPP3PP/2KR1B1R w - - 0 1");
        bench.push_back("3rr1k1/pp3pp1/1qn2np1/8/3p4/PP1R1P2/2P1NQPP/R1B3K1 b - - 0 1");
        bench.push_back("2r1nrk1/p2q1ppp/bp1p4/n1pPp3/P1P1P3/2PBB1N1/4QPPP/R4RK1 w - - 0 1");
        bench.push_back("r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1");
        bench.push_back("r2q1rk1/4bppp/p2p4/2pP4/3pP3/3Q4/PP1B1PPP/R3R1K1 w - - 0 1");
        bench.push_back("rnb2r1k/pp2p2p/2pp2p1/q2P1p2/8/1Pb2NP1/PB2PPBP/R2Q1RK1 w - - 0 1");
        bench.push_back("2r3k1/1p2q1pp/2b1pr2/p1pp4/6Q1/1P1PP1R1/P1PN2PP/5RK1 w - - 0 1");
        bench.push_back("r1bqkb1r/4npp1/p1p4p/1p1pP1B1/8/1B6/PPPN1PPP/R2Q1RK1 w kq - 0 1");
        bench.push_back("r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - - 0 1");
        bench.push_back("r1bq1rk1/pp2ppbp/2np2p1/2n5/P3PP2/N1P2N2/1PB3PP/R1B1QRK1 b - - 0 1");
        bench.push_back("3rr3/2pq2pk/p2p1pnp/8/2QBPP2/1P6/P5PP/4RRK1 b - - 90 90");
        bench.push_back("r4k2/pb2bp1r/1p1qp2p/3pNp2/3P1P2/2N3P1/PPP1Q2P/2KRR3 w - - 0 1");
        bench.push_back("3rn2k/ppb2rpp/2ppqp2/5N2/2P1P3/1P5Q/PB3PPP/3RR1K1 w - - 0 1");
        bench.push_back("2r2rk1/1bqnbpp1/1p1ppn1p/pP6/N1P1P3/P2B1N1P/1B2QPP1/R2R2K1 b - - 0 1");
        bench.push_back("r1bqk2r/pp2bppp/2p5/3pP3/P2Q1P2/2N1B3/1PP3PP/R4RK1 b kq - 0 1");
        bench.push_back("r2qnrnk/p2b2b1/1p1p2pp/2pPpp2/1PP1P3/PRNBB3/3QNPPP/5RK1 w - - 0 1");
        bench.push_back("qrb1nnkr/ppp2ppp/4pb2/3p4/2P5/1P3N2/P2PPPPP/QRBB1NKR w HBhb - 1 4 (FRC)");
        bench.push_back("rbknnrbq/1pppp1pp/8/p7/P3p3/2N5/1PPP1PPP/RBK1NRBQ w FAfa - 0 4 (FRC)");
        bench.push_back("1bbnrkqr/ppp2ppp/1n6/3pp3/7P/1N1P4/PPP1PPP1/1BBNRKQR w HEhe - 0 4 (FRC)");

        return bench;
    }

    int perft_tests()
    {
        // Store initial state
        bool Chess960 = UCI::Options::UCI_Chess960;

        // Loop over each position
        int n_failed = 0;
        auto tests = test_suite();
        for (auto& test : tests)
        {
            // Check if this is a FRC position
            UCI::Options::UCI_Chess960 = (test.fen().find("(FRC)") != std::string::npos);

            Position pos(test.fen());
            Histories hists;
            auto result = Search::perft<false>(pos, test.depth(), hists);
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

        UCI::Options::UCI_Chess960 = Chess960;
        std::cout << "\nFailed/total tests: " << n_failed << "/" << tests.size() << std::endl;
        return n_failed;
    }


    void bench(Search::Limits limits, int threads, int hash)
    {
        // Resize thread pool and hash
        pool->resize(threads);
        ttable.resize(hash);
        pool->clear();
        ttable.clear();

        // Start master timer
        Search::Timer time;
        uint64_t nodes = 0;

        // Store initial state
        bool Chess960 = UCI::Options::UCI_Chess960;

        // Loop over each position
        int i = 0;
        Position& pos = pool->position();
        std::vector<std::string> fens = bench_suite();
        for (auto fen : fens)
        {
            // Check if this is a FRC position
            UCI::Options::UCI_Chess960 = (fen.find("(FRC)") != std::string::npos);

            // Update position
            pos = Position(fen);
            pos.set_init_ply();
            pool->update_position_threads();
            std::cerr << "\nPosition " << (++i) << "/" << fens.size() << ": " << fen << std::endl;

            // Start searching and wait for completion
            Search::Timer pos_timer;
            pool->search(pos_timer, limits);
            pool->wait();

            // Update number of nodes
            nodes += pool->nodes_searched();
        }

        // Output bench stats
        double elapsed = time.elapsed();
        std::cerr << "\n---------------------------"                   << std::endl;
        std::cerr << "Nodes searched:   " << nodes                     << std::endl;
        std::cerr << "Elapsed time (s): " << std::setw(7) << elapsed   << std::endl;
        std::cerr << "Nodes per second: " << uint64_t(nodes / elapsed) << std::endl;

        // Restore initial options, thread pool and hash
        UCI::Options::UCI_Chess960 = Chess960;
        pool->resize(UCI::Options::Threads);
        ttable.resize(UCI::Options::Hash);
    }


    int legality_tests()
    {
        auto tests = test_suite();

        int n_failed = 0;
        for (auto& test : tests)
        {
            Position pos(test.fen());

            auto moves = pos.generate_moves(MoveGenType::LEGAL);
            int reference = moves.length();

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
