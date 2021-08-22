#include "Evaluation.hpp"
#include "Search.hpp"
#include "Tests.hpp"
#include "Transpositions.hpp"
#include "Types.hpp"
#include "UCI.hpp"
#include <iostream>
#include <sstream>
#include <string>


namespace UCI
{
	namespace Options
	{
		int hash = 16;
	}


	void main_loop()
	{
		std::string token;
		while (token != "quit")
		{
			std::string cmd;
			std::getline(std::cin, cmd);
			Stream stream(cmd);

			// Read the first token
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
				std::cout << Search::base_position->board() << std::endl;
			else if (token == "eval")
				evaluation(*Search::base_position, true);
			else if (token == "test")
			{
				int t1 = Tests::perft_tests();
				int t2 = Tests::perft_techniques_tests<false, true , false>();
				int t3 = Tests::perft_techniques_tests<true,  false, false>();
				int t4 = Tests::perft_techniques_tests<true,  true , false>();
				int t5 = Tests::perft_techniques_tests<false, false,  true>();

				std::cout << "\nTest summary" << std::endl;
				std::cout << "  Perft:        " << t1 << " failed cases" << std::endl;
				std::cout << "  TT:           " << t2 << " failed cases" << std::endl;
				std::cout << "  Orderer:      " << t3 << " failed cases" << std::endl;
				std::cout << "  TT + Orderer: " << t4 << " failed cases" << std::endl;
				std::cout << "  Legality:     " << t5 << " failed cases" << std::endl;
			}
		}
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
				Search::Parameters::multiPV = std::min(std::max(std::stoi(value), 1), 255);
			else if (name == "Threads")
				Search::Parameters::n_threads = std::min(std::max(std::stoi(value), 1), 512);
			else if (name == "Ponder")
				Search::Parameters::ponder = (value == "true");
			else
				std::cout << "Unknown option " << name << std::endl;
		}
	}



	void go(Stream& stream)
	{
		std::string token;
		int perft_depth = 0;
		auto& limits = Search::Parameters::limits;

		limits = Search::Limits();
		while (stream >> token)
			if (token == "searchmoves")
				while (stream >> token)
					limits.searchmoves.push_back(move_from_uci(*Search::base_position, token));
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
			Search::go_perft(perft_depth);
			return;
		}

		Search::end_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(INT32_MAX);
		Search::update_time();
		Search::start_search_threads();
	}



	void stop(Stream& stream)
	{
		Search::stop_search_threads();
	}



	void quit(Stream& stream)
	{
		Search::stop_search_threads();
	}



	void position(Stream& stream)
	{
		Position& pos = *Search::base_position;

		std::string token;
		stream >> token;

		if (token == "startpos")
		{
			pos = Position();

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
	}



	void ponderhit(Stream& stream)
	{
		Search::Parameters::limits.ponder = false;
		Search::update_time();
	}



	void ucinewgame(Stream& stream)
	{
		ttable.clear();
	}



	void isready(Stream& stream)
	{
		// Mandatory readyok output when all set
		std::cout << "readyok" << std::endl;
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