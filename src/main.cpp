#include "Types.hpp"
#include "Position.hpp"
#include "Tests.hpp"
#include "Zobrist.hpp"
#include "Search.hpp"
#include "UCI.hpp"
#include "Endgame.hpp"
#include <chrono>
#include <memory>

int main()
{
    init_types();
    Bitboards::init_bitboards();
    Zobrist::build_rnd_hashes();
    ttable = TranspositionTable<TranspositionEntry>(16);
    Search::base_position = std::make_unique<Position>();

    UCI::main_loop();
}
