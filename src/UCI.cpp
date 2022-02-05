#include "Evaluation.hpp"
#include "Search.hpp"
#include "Tests.hpp"
#include "Thread.hpp"
#include "Transpositions.hpp"
#include "Types.hpp"
#include "UCI.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>


namespace UCI
{
    namespace Options
    {
        int hash = 16;
    }


    void main_loop(std::string args)
    {
        std::string token;
        while (token != "quit")
        {
            std::string cmd;
            if (args == "")
                std::getline(std::cin, cmd);
            else
                cmd = args;
            Stream stream(cmd);

            // Read the first token
            token.clear();
            stream >> token;

            // Switch on the received token
            if (token == "quit")
                quit(stream);
            else if (token == "stop")
                stop(stream);
            else if (token == "uci")
                uci(stream);
            else if (token == "setoption")
                setoption(stream);
            else if (token == "isready")
                isready(stream);
            else if (token == "ucinewgame")
                ucinewgame(stream);
            else if (token == "go")
                go(stream);
            else if (token == "position")
                position(stream);
            else if (token == "ponderhit")
                ponderhit(stream);

            // Non-UCI commands
            else if (token == "board")
                std::cout << pool->position().board() << std::endl;
            else if (token == "eval")
                evaluation(pool->position(), true);
            else if (token == "test")
                test();
            else if (token == "perft")
                perft(stream);
            else if (token == "tt")
                tt_query();
            else if (token == "bench")
                bench(stream);

            // Unknown commands
            else if (token != "")
                std::cout << "Unknown command " << token << std::endl;

            // Quit if a single command has been passed
            if (args != "")
                break;
        }

        // Ensure search threads are stopped on exit
        pool->kill_threads();
    }



    void uci(Stream& stream)
    {
        std::cout << "id name pawn" << std::endl;
        std::cout << "id author ruicoelhopedro" << std::endl;

        // Send options
        std::cout << std::endl;
        std::cout << "option name Hash type spin default 16 min 1 max " << ttable.max_size() << std::endl;
        std::cout << "option name MultiPV type spin default 1 min 1 max 255" << std::endl;
        std::cout << "option name Ponder type check default false" << std::endl;
        std::cout << "option name Threads type spin default 1 min 1 max 512" << std::endl;

        // Mandatory uciok at the end
        std::cout << "uciok" << std::endl;
    }



    void setoption(Stream& stream)
    {
        std::string token;
        stream >> token;
        if (stream && token == "name")
        {
            // Read option name (can contain spaces)
            std::string name;
            while (stream >> token && token != "value")
                name += name.empty() ? token : (" " + token);

            // Read value (can also contain spaces)
            std::string value;
            while (stream >> token)
                value += value.empty() ? token : (" " + token);

            // Set value
            if (name == "Hash")
            {
                Options::hash = std::min(std::max(std::stoi(value), 1), ttable.max_size());
                ttable.resize(Options::hash);
            }
            else if (name == "MultiPV")
            {
                Search::Parameters::multiPV = std::min(std::max(std::stoi(value), 1), 255);
                pool->update_multiPV(Search::Parameters::multiPV);
            }
            else if (name == "Threads")
                pool->resize(std::min(std::max(std::stoi(value), 1), 512));
            else if (name == "Ponder")
                Search::Parameters::ponder = (value == "true");
            else
                std::cout << "Unknown option " << name << std::endl;
        }
    }



    void go(Stream& stream)
    {
        std::string token;
        Search::Time time;
        Search::Limits limits;

        limits = Search::Limits();
        while (stream >> token)
            if (token == "searchmoves")
                while (stream >> token)
                    limits.searchmoves.push(move_from_uci(pool->position(), token));
            else if (token == "wtime")
                stream >> limits.time[WHITE];
            else if (token == "btime")
                stream >> limits.time[BLACK];
            else if (token == "winc")
                stream >> limits.incr[WHITE];
            else if (token == "binc")
                stream >> limits.incr[BLACK];
            else if (token == "movestogo")
                stream >> limits.movestogo;
            else if (token == "depth")
                stream >> limits.depth;
            else if (token == "nodes")
                stream >> limits.nodes;
            else if (token == "movetime")
                stream >> limits.movetime;
            else if (token == "mate")
                stream >> limits.mate;
            else if (token == "infinite")
                limits.infinite = true;
            else if (token == "ponder")
                limits.ponder = true;

        Search::update_time(pool->position(), limits, time);
        pool->search(time, limits);
    }



    void stop(Stream& stream)
    {
        pool->stop();
    }



    void quit(Stream& stream)
    {
    }



    void position(Stream& stream)
    {
        Position& pos = pool->position();

        std::string token;
        stream >> token;

        if (token == "startpos")
        {
            pos.reset_startpos();

            // Check if moves token has been passed
            if (stream >> token && token != "moves")
                return;
        }
        else if (token == "fen")
        {
            // Build FEN string
            std::string fen;
            while (stream >> token && token != "moves")
                fen += token + " ";

            pos.update_from(fen);
        }
        else
            return;

        // Push moves to the position
        Move move;
        while (stream >> token && (move = move_from_uci(pos, token)) != MOVE_NULL)
            pos.make_move(move);

        // After all moves are pushed, prepare the position for search
        pos.prepare();
        pool->update_position_threads();
    }



    void ponderhit(Stream& stream)
    {
        pool->ponderhit();
    }



    void ucinewgame(Stream& stream)
    {
        ttable.clear();
        //Search::set_num_threads(Search::Parameters::n_threads);
    }



    void isready(Stream& stream)
    {
        // Mandatory readyok output when all set
        std::cout << "readyok" << std::endl;
    }



    void test()
    {
        constexpr int NUM_TESTS = 7;
        int results[NUM_TESTS];

        results[0] = Tests::perft_tests();
        results[1] = Tests::perft_techniques_tests<2, false, false, false, true>();
        results[2] = Tests::perft_techniques_tests<2, false,  true, false, false>();
        results[3] = Tests::perft_techniques_tests<2,  true, false, false, false>();
        results[4] = Tests::perft_techniques_tests<2,  true,  true, false, false>();
        results[5] = Tests::perft_techniques_tests<3, false, false,  true, false>();
        results[6] = Tests::fen_tests();

        std::cout << "\nTest summary" << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::cout << "  Perft:        " << std::setw(4) << results[0] << " failed cases" << std::endl;
        std::cout << "  Validity:     " << std::setw(4) << results[1] << " failed cases" << std::endl;
        std::cout << "  TT:           " << std::setw(4) << results[2] << " failed cases" << std::endl;
        std::cout << "  Orderer:      " << std::setw(4) << results[3] << " failed cases" << std::endl;
        std::cout << "  TT + Orderer: " << std::setw(4) << results[4] << " failed cases" << std::endl;
        std::cout << "  Legality:     " << std::setw(4) << results[5] << " failed cases" << std::endl;
        std::cout << "  FEN Parsing:  " << std::setw(4) << results[6] << " failed cases" << std::endl;
        std::cout << "---------------------------------" << std::endl;

        // Final test results
        for (int i = 0; i < NUM_TESTS; i++)
            if (results[i] > 0)
            {
                std::cout << "Tests failed" << std::endl;
                return;
            }

        std::cout << "Tests passed" << std::endl;
    }



    void perft(Stream& stream)
    {
        int depth;
        if (stream >> depth)
        {
            int64_t nodes = Search::perft<true>(pool->position(), depth);
            std::cout << "\nNodes searched: " << nodes << std::endl;
        }
    }



    void tt_query()
    {
        const Position& pos = pool->position();
        TranspositionEntry* entry;
        if (ttable.query(pos.hash(), &entry))
        {
            if (entry->is_empty())
            {
                std::cout << "Empty entry" << std::endl;
            }
            else
            {
                Color color = turn_to_color(pos.board().turn());
                std::cout << "--------------------------------" << std::endl;
                std::cout << " Score:        ";
                if (entry->type() != EntryType::EXACT)
                    std::cout << (entry->type() == EntryType::LOWER_BOUND ? ">= " : "<= ");
                std::cout << color * entry->score() << " (White)" << std::endl;
                std::cout << " Depth:        " << static_cast<int>(entry->depth()) << std::endl;
                std::cout << " Move:         " << entry->hash_move() << std::endl;
                std::cout << " Static eval:  " << color * entry->static_eval() << " (White)" << std::endl;
                std::cout << " Hash:         " << std::hex << entry->hash() << std::dec << std::endl;
                std::cout << "--------------------------------" << std::endl;
            }
        }
        else
        {
            std::cout << "No TT entry for this position" << std::endl;
        }
        std::cout << std::endl;
    }



    void bench(Stream& stream)
    {
        auto tests = Tests::bench_suite();
        Depth depth = 11;
        int token;
        if (stream >> token)
            depth = token;
        Tests::bench_tests(depth, tests);
    }



    Move move_from_uci(Position& position, std::string move_str)
    {
        // Get moves for current position
        auto list = position.generate_moves(MoveGenType::LEGAL);

        // Search for the move in the move list
        for (auto& move : list)
            if (move.to_uci() == move_str)
                return move;

        return MOVE_NULL;
    }
}
