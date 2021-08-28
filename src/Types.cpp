#include "Types.hpp"
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


std::vector<std::string> split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        if (token.length() > 0)
            tokens.push_back(token);
    }
    return tokens;
}
