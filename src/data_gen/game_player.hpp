#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include "data_gen/data_gen.hpp"
#include <string>
#include <vector>


namespace GamePlayer
{
    void play_games(std::istringstream& stream);

    void games_to_epd(std::istringstream& stream);
}