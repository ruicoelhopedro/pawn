#pragma once

#include <iostream>
#include <sstream>
#include <string>

using Stream = std::istringstream;

namespace UCI
{
    namespace Options
    {
        extern int hash;
    }

    void main_loop(std::string args = "");
    void setoption(Stream& stream);
    void uci(Stream& stream);
    void go(Stream& stream);
    void stop(Stream& stream);
    void quit(Stream& stream);
    void position(Stream& stream);
    void ponderhit(Stream& stream);
    void ucinewgame(Stream& stream);
    void isready(Stream& stream);

    void test();
    void perft(Stream& stream);
    void tt_query();
    void bench(Stream& stream);

    Move move_from_uci(Position& position, std::string move_str);
}
