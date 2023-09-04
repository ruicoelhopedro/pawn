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
        int depth = 9;
        std::size_t runs_per_fen = 1000000;
        int adjudication = SCORE_MATE_FOUND;
        std::string book = "";
        std::string output_file = "output.dat";
        uint random_probability = 25;
        int store_min_ply = 15;
        int seed = 0;
        bool shallow_depth_pruning = false;

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
            else if (token == "random_probability")
                stream >> random_probability;
            else if (token == "store_min_ply")
                stream >> store_min_ply;
            else if (token == "seed")
                stream >> seed;
            else if (token == "shallow_depth_pruning")
                shallow_depth_pruning = true;
            else
            {
                std::cout << "Unknown option " << token << std::endl;
                return;
            }

        // Minimal sanity checks
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

        // Store and reset shallow depth pruning
        bool sdp = UCI::Options::ShallowDepthPruning;
        UCI::Options::ShallowDepthPruning = shallow_depth_pruning;

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

                // Position warmup stage: the initial position for the game is obtained by a
                // mix of search-based and random moves from the list of given FEN positions
                bool valid_game = true;
                for (int ply = 0; ply < store_min_ply; ply++)
                {
                    // Search or pick a random move?
                    Move move;
                    if (random.next(100) < random_probability)
                    {
                        // Random mover: generate legal moves for this position
                        Move moves[NUM_MAX_MOVES];
                        MoveList move_list(moves);
                        pos.board().generate_moves(move_list, MoveGenType::LEGAL);

                        // Pick a move
                        int num_moves = move_list.length();
                        move = num_moves > 0 ? moves[random.next(num_moves)] : MOVE_NULL;
                    }
                    else
                    {
                        // Search-based: pick the bestmove for this position
                        move = thread.simple_search(pos, limits).bestmove;
                    }

                    // Check if the game ended
                    valid_game = move != MOVE_NULL;
                    if (!valid_game)
                        break;

                    // Prepare next iteration
                    pos.make_move(move);
                    pos.set_init_ply();
                }

                // Is the reached position usable?
                SearchResult result = thread.simple_search(pos, limits);
                if (!valid_game || result.bestmove == MOVE_NULL)
                    continue;

                // Register a new game
                BinaryGame game;
                game.begin(pos.board());

                // Game loop
                while (result.bestmove != MOVE_NULL)
                {
                    game.push(result.bestmove, result.score);
                    pos.make_move(result.bestmove);
                    pos.set_init_ply();
                    result = thread.simple_search(pos, limits);
                }

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

        // Restore shallow depth pruning
        UCI::Options::ShallowDepthPruning = sdp;
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


    bool file_valid(std::string filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
            return false;

        BinaryGame game;
        std::size_t game_number = 0;
        while(BinaryGame::read(file, game))
        {
            game_number++;
            std::size_t move_number = 0;

            // Loop over all moves ensuring they are legal
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
            {
                move_number++;
                if (node.move != MOVE_NULL)
                {
                    if(!board.legal(node.move))
                    {
                        std::cerr << "Game " << game_number << " (move " << move_number << "): "
                                  << "Illegal move " << node.move
                                  << " in position " << board.to_fen()
                                  << std::endl;
                        return false;
                    }
                    board = board.make_move(node.move);
                }
                // Ensure the last score is the game outcome (-1, 0 or 1)
                else if (abs(node.score) > 1)
                {
                    std::cerr << "Game " << game_number << ": "
                              << "Bad game termination: " << node.score
                              << std::endl;
                    return false;
                }
            }
        }

        // Ensure the file at the end
        if (!file.eof())
        {
            std::cerr << "Bad EOF" << std::endl;
            return false;
        }

        return true;
    }


    void check_games(std::istringstream& stream)
    {
        // Files to parse
        std::string token;
        std::vector<std::string> files;
        while (stream >> token)
            files.push_back(token);
        assert(files.size() > 0 && "No input files have been passed!");

        // Parse each file
        for (auto filename : files)
            std::cout << (file_valid(filename) ? " [ OK ] " : " [FAIL] ")
                      << filename
                      << std::endl;
    }
}