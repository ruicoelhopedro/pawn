#include "Evaluation.hpp"
#include "Search.hpp"
#include "Tests.hpp"
#include "Hash.hpp"
#include "Types.hpp"
#include "UCI.hpp"
#include "Thread.hpp"
#include "syzygy/syzygy.hpp"
#include <array>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>


namespace UCI
{
#ifdef GIT_VERSION
    const std::string VERSION = "pawn " + std::string(GIT_VERSION);
#else
    const std::string VERSION = "pawn v4.0";
#endif


    std::unique_ptr<ThreadPool> pool;
    std::map<std::string, Option, OptionNameCompare> OptionsMap;


    namespace Options
    {
        int Hash;
        int MultiPV;
        bool Ponder;
        bool UCI_Chess960;
        int Threads;
        int MoveOverhead;
        std::string NNUE_File;
        std::string TB_Path;
        int TB_ProbeDepth;
        int TB_ProbeLimit;
    }


    void Option::set(std::string value)
    {
        if (m_type == OptionType::CHECK)
        {
            *std::get<bool*>(m_data) = (value == "true");
        }
        else if (m_type == OptionType::SPIN)
        {
            *std::get<int*>(m_data) = std::clamp(std::stoi(value), m_min, m_max);
        }
        else if (m_type == OptionType::COMBO)
        {
            auto val = std::find(m_var.begin(), m_var.end(), value);
            if (val != m_var.end())
                *std::get<std::string*>(m_data) = *val;
        }
        else if (m_type == OptionType::STRING)
        {
            *std::get<std::string*>(m_data) = value;
        }

        // On-change callback
        if (std::visit([](auto func){ return func != nullptr; }, m_change))
        {
            if (m_type == CHECK)
                std::get<OnChange<bool>>(m_change)(*std::get<bool*>(m_data));
            else if (m_type == SPIN)
                std::get<OnChange<int>>(m_change)(*std::get<int*>(m_data));
            else if (m_type == BUTTON)
                std::get<OnChange<>>(m_change)();
            else
                std::get<OnChange<std::string>>(m_change)(*std::get<std::string*>(m_data));
        }
    }



    std::ostream& operator<<(std::ostream& out, const Option& option)
    {
        std::array<std::string, 5> types{ "check", "spin", "combo", "button", "string" };
        out << " type " << types[option.m_type];
        
        // Button type has no default
        if (option.m_type != BUTTON)
        {
            out << " default ";
            // Ensure we print true or false for check types
            if (option.m_type == CHECK)
                out << (std::get<bool>(option.m_default) ? "true" : "false");
            else
                std::visit([&out](auto value){ out << value; }, option.m_default);

            // Spin has min and max
            if (option.m_type == SPIN)
                out << " min " << option.m_min << " max " << option.m_max;
            // Combo has the list of vars
            else if (option.m_type == COMBO)
                for (std::string var : option.m_var)
                    out << " var " << var;
        }

        return out;
    }



    void init()
    {
        // Default values
        constexpr std::size_t n_threads = 1;
        constexpr std::size_t hash_size = 16;

        // Create thread pool
        pool = std::make_unique<ThreadPool>(n_threads, hash_size);

        // Initialise options
        OptionsMap.emplace("Clear Hash",    Option(OnChange<>([]() { pool->tt().clear(); })));
        OptionsMap.emplace("Hash",          Option(&Options::Hash, hash_size, 1, pool->tt().max_size(),
                                                   [](int v) { pool->tt().resize(v); }));
        OptionsMap.emplace("MultiPV",       Option(&Options::MultiPV, 1, 1, 255));
        OptionsMap.emplace("Threads",       Option(&Options::Threads, n_threads, 1, 512,
                                                   [](int v) { pool->resize(v); }));
        OptionsMap.emplace("Move Overhead", Option(&Options::MoveOverhead, 0, 0, 5000));
        OptionsMap.emplace("Ponder",        Option(&Options::Ponder, false));
        OptionsMap.emplace("UCI_Chess960",  Option(&Options::UCI_Chess960, false));
        OptionsMap.emplace("NNUE_File",     Option(&Options::NNUE_File, "", NNUE::load));
        OptionsMap.emplace("SyzygyPath",    Option(&Options::TB_Path, "", Syzygy::load));
        OptionsMap.emplace("SyzygyProbeDepth", Option(&Options::TB_ProbeDepth, 1, 1, 199));
        OptionsMap.emplace("SyzygyProbeLimit", Option(&Options::TB_ProbeLimit, 7, 0, 7, [](int v)
        {
            Syzygy::Cardinality = std::min(Syzygy::Cardinality, v);
        }));
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
            stream >> token;

            // Switch on the received token
            if (token == "quit")
                quit();
            else if (token == "stop")
                stop();
            else if (token == "uci")
                uci();
            else if (token == "setoption")
                setoption(stream);
            else if (token == "isready")
                isready();
            else if (token == "ucinewgame")
                ucinewgame();
            else if (token == "go")
                go(stream);
            else if (token == "position")
                position(stream);
            else if (token == "ponderhit")
                ponderhit();

            // Non-UCI commands
            else if (token == "board")
                std::cout << pool->position().board() << std::endl;
            else if (token == "eval")
                Evaluation::eval_table(pool->position().board());
            else if (token == "bench")
                bench(stream);
            else if (token == "test")
            {
                int t1 = Tests::perft_tests();
                int t2 = Tests::perft_techniques_tests<true, false, false>();
                int t3 = Tests::perft_techniques_tests<false, true, false>();
                int t4 = Tests::perft_techniques_tests<false, true,  true>();

                std::cout << "\nTest summary" << std::endl;
                std::cout << "  Perft:        " << t1 << " failed cases" << std::endl;
                std::cout << "  Orderer:      " << t2 << " failed cases" << std::endl;
                std::cout << "  Legality:     " << t3 << " failed cases" << std::endl;
                std::cout << "  Validity:     " << t4 << " failed cases" << std::endl;

                if (t1 + t2 + t3 + t4 == 0)
                    std::cout << "\nAll tests passed" << std::endl;
            }

            // Unknown command
            else if (token != "")
                std::cout << "Unknown command " << token << std::endl;
        
            // Quit if a single command has been passed
            if (args != "")
                break;
        }

        // Wait for completion, then kill threads
        pool->wait();
        pool->kill_threads();
    }



    void uci()
    {
        std::cout << "id name " << VERSION << std::endl;
        std::cout << "id author ruicoelhopedro" << std::endl;

        // Send options
        std::cout << std::endl;
        for (const auto &[name, option] : OptionsMap)
            std::cout << "option name " << name << option << std::endl;

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

            if (OptionsMap.find(name) != OptionsMap.end())
                OptionsMap.at(name).set(value);
            else
                std::cout << "Unknown option " << name << std::endl;
        }
    }



    void go(Stream& stream)
    {
        std::string token;
        int perft_depth = 0;
        Search::Timer timer;
        Search::Limits limits;

        while (stream >> token)
            if (token == "searchmoves")
                while (stream >> token)
                    limits.searchmoves.push_back(move_from_uci(pool->position(), token));
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
            else if (token == "perft")
                stream >> perft_depth;

        // Check if perft search
        if (perft_depth > 0)
        {
            std::unique_ptr<Histories> hists;
            int64_t nodes = Search::perft<true>(pool->position(), perft_depth, *hists);
            std::cout << "\nNodes searched: " << nodes << std::endl;
            return;
        }

        pool->search(timer, limits);
    }



    void stop()
    {
        pool->stop();
    }



    void quit()
    {
        pool->stop();
    }



    void position(Stream& stream)
    {
        Position& pos =  pool->position();

        std::string token;
        stream >> token;

        if (token == "startpos")
        {
            pos = Position();

            // Consume moves token, if passed
            while (stream >> token && token != "moves");
        }
        else if (token == "fen")
        {
            // Build FEN string
            std::string fen;
            while (stream >> token && token != "moves")
                fen += token + " ";
            pos = Position(fen);
        }
        else
            return;

        // Push moves to the position
        Move move;
        while (stream >> token && (move = move_from_uci(pos, token)) != MOVE_NULL)
        {
            pos.make_move(move);
            pos.set_init_ply();
        }

        pool->update_position_threads();
    }



    void ponderhit()
    {
        pool->ponderhit();
    }



    void ucinewgame()
    {
        pool->clear();
    }



    void isready()
    {
        // Mandatory readyok output when all set
        std::cout << "readyok" << std::endl;
    }



    void bench(Stream& stream)
    {
        Depth depth = 13;
        int threads = 1;
        int hash = 16;
        
        std::string token;
        if (stream >> token)
            depth = std::stoi(token);
        if (stream >> token)
            threads = std::stoi(token);
        if (stream >> token)
            hash = std::stoi(token);

        Search::Limits limits;
        limits.depth = depth;

        Tests::bench(limits, threads, hash);
    }



    Move move_from_uci(Position& position, std::string move_str)
    {
        // Get moves for current position
        auto list = position.generate_moves(MoveGenType::LEGAL);

        // Search for the move in the move list
        for (auto& move : list)
            if (position.board().to_uci(move) == move_str)
                return move;

        return MOVE_NULL;
    }


    bool OptionNameCompare::operator()(const std::string& a, const std::string& b) const
    {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
                                            [](char c1, char c2)
                                            {
                                                return tolower(c1) < tolower(c2);
                                            });
    }
}
