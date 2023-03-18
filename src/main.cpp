#include "Types.hpp"
#include "Position.hpp"
#include "PieceSquareTables.hpp"
#include "Tests.hpp"
#include "Zobrist.hpp"
#include "Search.hpp"
#include "UCI.hpp"
#include "Thread.hpp"
#include <chrono>
#include <sstream>

int main(int argc, char** argv)
{
    Bitboards::init_bitboards();
    Zobrist::build_rnd_hashes();
    PSQT::init();
    Tune::init();
    UCI::init_options();
    ttable = HashTable<TranspositionEntry>(16);
    pool = new ThreadPool();

    // Handle passed arguments
    std::stringstream ss;
    if (argc > 1)
        for (int i = 1; i < argc; i++)
            ss << argv[i] << ' ';

    UCI::main_loop(ss.str());

    pool->kill_threads();
    PSQT::clean();
}
