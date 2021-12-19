#include "Types.hpp"
#include "Position.hpp"
#include "Tests.hpp"
#include "Zobrist.hpp"
#include "Search.hpp"
#include "UCI.hpp"
#include "Endgame.hpp"
#include <chrono>
#include <sstream>

int main(int argc, char** argv)
{
    init_types();
    Bitboards::init_bitboards();
    Zobrist::build_rnd_hashes();
    ttable = TranspositionTable<TranspositionEntry>(16);
    Search::base_position = new Position();
    Search::set_num_threads(1);

    // Handle passed arguments
    std::stringstream ss;
    if (argc > 1)
        for (int i = 1; i < argc; i++)
            ss << argv[i] << ' ';

    UCI::main_loop(ss.str());

    Search::kill_search_threads();
}
