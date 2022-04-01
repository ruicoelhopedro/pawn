#include "Types.hpp"
#include "Position.hpp"
#include "Tests.hpp"
#include "Zobrist.hpp"
#include "Search.hpp"
#include "UCI.hpp"
#include "Thread.hpp"
#include <chrono>

int main()
{
    Bitboards::init_bitboards();
    Zobrist::build_rnd_hashes();
    UCI::init_options();
    ttable = HashTable<TranspositionEntry>(16);
    pool = new ThreadPool();

    UCI::main_loop();
    pool->kill_threads();
}
