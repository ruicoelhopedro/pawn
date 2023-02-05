#include "Types.hpp"
#include <array>
#include <sstream>
#include <string>
#include <vector>


std::string get_square(Square square)
{
    constexpr char ranks[8] = { '1', '2', '3', '4', '5', '6', '7', '8' };
    constexpr char files[8] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h' };
    char name[2] = { files[file(square)], ranks[rank(square)] };
    return std::string(name, 2);
}

std::array<Debug::Entry, Debug::NUM_DEBUG_SLOTS> Debug::debug_slots;
