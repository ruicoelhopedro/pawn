#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/fen_score.hpp"
#include <string>


namespace GamePlayer
{
    void play_games(std::istringstream& stream)
    {
        // Parameters and their default values
        int depth = 7;
        std::size_t runs_per_fen = 1000000;
        int adjudication = 4000;
        std::string book = "";
        std::string output_file = "output.dat";
        int random_min_ply = 0;
        int random_max_ply = 20;
        int random_max_count = 5;
        uint random_probability = 25;
        int store_min_ply = 15;
        int seed = 0;

        // Read passed parameters
        std::string token;
        while (stream >> token)
            if (token == "depth")
                stream >> depth;
            else if (token == "runs_per_fen")
                stream >> runs_per_fen;
            else if (token == "adjudication")
                stream >> adjudication;
            else if (token == "book")
                stream >> book;
            else if (token == "output_file")
                stream >> output_file;
            else if (token == "random_min_ply")
                stream >> random_min_ply;
            else if (token == "random_max_ply")
                stream >> random_max_ply;
            else if (token == "random_max_count")
                stream >> random_max_count;
            else if (token == "random_probability")
                stream >> random_probability;
            else if (token == "store_min_ply")
                stream >> store_min_ply;
            else if (token == "seed")
                stream >> seed;
            else
            {
                std::cout << "Unknown option " << token << std::endl;
                return;
            }

        // Minimal sanity checks
        random_min_ply = std::max(1, random_min_ply);
        store_min_ply = std::max(1, store_min_ply);

        // Build list of FENs to use
        std::vector<std::string> fens;
        if (book == "")
            fens.push_back(Board().to_fen());
        else
            fens = read_fens(book);

        // Init PRNG
        PseudoRandom random(seed);

        // Open output file
        std::ofstream output(output_file);
        assert(output.is_open() && "Failed to open output file!");

        // Current and total number of games to run
        std::size_t n_games_completed = 0;
        std::size_t n_total_games = fens.size() * runs_per_fen;

        // Prepare search data
        Search::Limits limits;
        limits.depth = depth;
        Thread& thread = pool->front();

        // Loop over each FEN
        for (std::size_t i_fen = 0; i_fen < fens.size(); i_fen++)
        {
            const std::string& fen = fens[i_fen];

            // Runs loop
            for (std::size_t i_run = 0; i_run < runs_per_fen; i_run++)
            {
                // Initialise position and clear search data
                Position pos(fen);
                pos.set_init_ply();
                UCI::ucinewgame(stream);

                // Register a new game
                BinaryGame game;
                int n_random_moves = 0;

                // Game loop
                GamePosition state = thread.simple_search(pos, limits);
                while (state.bestmove != MOVE_NULL && abs(state.score) < adjudication)
                {
                    int ply = pos.game_ply();

                    // Select the move that we will play
                    Move move = state.bestmove;

                    // Check if we are going to play a random move this time
                    if (n_random_moves < random_max_count &&
                        ply >= random_min_ply &&
                        ply <= random_max_ply &&
                        random.next(100) < random_probability)
                    {
                        n_random_moves++;

                        // Generate legal moves for this position
                        Move moves[NUM_MAX_MOVES];
                        MoveList move_list(moves);
                        pos.board().generate_moves(move_list, MoveGenType::LEGAL);

                        // Pick a move
                        move = move_list[random.next(move_list.length())];
                    }

                    // Start storing the game if we have just reached the starting point
                    if (ply == store_min_ply)
                        game.begin(pos.board());

                    // Store this node
                    if (ply >= store_min_ply)
                        game.push(move, state.score);

                    // Prepare next iteration
                    pos.make_move(move);
                    pos.set_init_ply();
                    state = thread.simple_search(pos, limits);
                }

                // If the game was adjudicated, write the last node
                if (state.bestmove != MOVE_NULL)
                    game.push(state.bestmove, state.score);

                // Game is completed, write to the output file
                game.write(output);

                // Report progress
                n_games_completed++;
                std::cout << "\r"
                          << "Games: " << n_games_completed << "/" << n_total_games << " "
                          << "FENs: "  << (i_fen + 1) << "/" << fens.size()         << " "
                          << "Runs: "  << (i_run + 1) << "/" << runs_per_fen
                          << std::flush;
            }
        }
        std::cout << std::endl;
    }


    void games_to_epd(std::istringstream& stream)
    {
        // Parameters
        std::string input_file_path;
        std::string output_file_path;

        // Parameter reading
        assert((stream >> input_file_path) && "Input file path required!");
        assert((stream >> output_file_path) && "Output file path required!");

        // Open files
        std::ifstream ifile(input_file_path);
        std::ofstream ofile(output_file_path);
        assert(ifile.is_open() && "Failed to open input file!");
        assert(ofile.is_open() && "Failed to open output file!");

        // Read entire file
        BinaryGame game;
        while(BinaryGame::read(ifile, game))
        {
             // Write each position, score and move
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                {
                    ofile << board.to_fen() << " " << node.move << " " << node.score << "\n";
                    // Make the move
                    assert(board.legal(node.move) && "Illegal move!");
                    board = board.make_move(node.move);
                }
        }
    }
}