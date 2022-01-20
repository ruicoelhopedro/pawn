#include "Types.hpp"
#include "Position.hpp"
#include "Tests.hpp"
#include "Zobrist.hpp"
#include "Search.hpp"
#include "UCI.hpp"
#include <chrono>

int main()
{
    Bitboards::init_bitboards();
    Zobrist::build_rnd_hashes();
    ttable = TranspositionTable<TranspositionEntry>(16);
    Search::base_position = new Position();
    Search::position_threads.push_back(*Search::base_position);

    UCI::main_loop();
}
