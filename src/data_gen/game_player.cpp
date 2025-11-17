#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "syzygy/syzygy.hpp"
#include "data_gen/game_player.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/fen_score.hpp"
#include <mutex>
#include <thread>
#include <string>


namespace GamePlayer
{
    void play_games(std::istringstream& stream)
    {
        // Parameters and their default values
        int depth = 8;
        int nodes = 0;
        std::size_t max_num_games = 0;
        std::string book = "";
        std::string output_file = "output.bin";
        int store_min_ply = 8;
        int seed = 0;
        int threads = 1;
        int hash = 1;
        int accept_threshold = 50;
        bool shallow_depth_pruning = false;
        std::string syzygy_path = "";
        bool chess_960 = false;

        // Read passed parameters
        std::string token;
        while (stream >> token)
            if (token == "depth")
                stream >> depth;
            else if (token == "nodes")
                stream >> nodes;
            else if (token == "max_num_games")
                stream >> max_num_games;
            else if (token == "book")
                stream >> book;
            else if (token == "output_file")
                stream >> output_file;
            else if (token == "store_min_ply")
                stream >> store_min_ply;
            else if (token == "seed")
                stream >> seed;
            else if (token == "threads")
                stream >> threads;
            else if (token == "hash")
                stream >> hash;
            else if (token == "accept_threshold")
                stream >> accept_threshold;
            else if (token == "shallow_depth_pruning")
                shallow_depth_pruning = true;
            else if (token == "syzygy_path")
                stream >> syzygy_path;
            else if (token == "chess_960")
                chess_960 = true;
            else
            {
                std::cout << "Unknown option " << token << std::endl;
                return;
            }

        // Minimal sanity checks
        store_min_ply = std::max(1, store_min_ply);

        // Load Syzygy tablebases
        if (syzygy_path != "")
            Syzygy::load(syzygy_path);

        // Update Chess960 status
        bool init_chess960 = UCI::Options::UCI_Chess960;
        UCI::Options::UCI_Chess960 = chess_960;

        // Build list of FENs to use
        std::vector<std::string> fens;
        if (book == "")
            fens.push_back(Board().to_fen());
        else
            fens = read_fens(book);

        // Open output file
        std::ofstream output(output_file);
        assert(output.is_open() && "Failed to open output file!");

        // Current number of games
        std::size_t n_games_completed = 0;

        // Prepare search data
        Search::Limits limits;
        if (depth > 0)
            limits.depth = depth;
        if (nodes > 0)
            limits.nodes = nodes;
        std::atomic_bool stop = false;
        ThreadSafeQueue<BinaryGame> queue;
        std::atomic_int n_active_threads = threads;
        auto start = std::chrono::steady_clock::now();

        // Store and reset shallow depth pruning
        bool sdp = UCI::Options::ShallowDepthPruning;
        UCI::Options::ShallowDepthPruning = shallow_depth_pruning;

        auto thread_loop = [=, &queue, &n_active_threads, &stop](std::size_t id)
        {
            // Init PRNG
            PseudoRandom random(seed + PseudoRandom::get(id));

            // Prepare search thread
            ThreadPool pool(1, hash);
            Thread& thread = pool.front();

            // Data generation loop
            while (!stop.load(std::memory_order_relaxed))
            {
                // Pick a random FEN
                const std::string& fen = fens[random.next(fens.size())];
                Position pos(fen);
                SearchResult result;

                // Initial position generation
                while (true)
                {
                    // Initialise position and clear search data
                    pos = Position(fen);
                    pos.set_init_ply();
                    pool.clear();

                    // Position warmup stage: the initial position for the game is obtained by a
                    // set of random moves from the list of given FEN positions
                    bool valid_game = true;
                    int num_plies = store_min_ply + random.next(2);
                    for (int ply = 0; ply < num_plies; ply++)
                    {
                        // Random mover: generate legal moves for this position
                        Move moves[NUM_MAX_MOVES];
                        MoveList move_list(moves);
                        pos.board().generate_moves(move_list, MoveGenType::LEGAL);

                        // Pick a move
                        int num_moves = move_list.length();
                        Move move = num_moves > 0 ? moves[random.next(num_moves)] : MOVE_NULL;

                        // Check if the game ended
                        valid_game = move != MOVE_NULL;
                        if (!valid_game)
                            break;

                        // Prepare next iteration
                        pos.make_move(move);
                        pos.set_init_ply();
                    }

                    // Is the reached position usable?
                    if (!valid_game)
                        continue;

                    // Determine if we keep using this position or discard it based on the score
                    result = thread.simple_search(pos, limits, threads > 1);
                    int prob = 200 * (1 + accept_threshold) / (1 + abs(result.score));
                    if (result.bestmove != MOVE_NULL && int(random.next(100)) <= prob)
                        break;
                }

                // Register a new game
                BinaryGame game;
                game.begin(pos.board());

                // Game loop
                while (result.bestmove != MOVE_NULL)
                {
                    game.push(result.bestmove, result.score);
                    pos.make_move(result.bestmove);
                    pos.set_init_ply();
                    result = thread.simple_search(pos, limits, threads > 1);
                }

                // Add termination node
                game.push(MOVE_NULL, result.score > 0 ? WHITE_COLOR : result.score < 0 ? BLACK_COLOR : NO_COLOR);
                queue.push(game);
            }
            n_active_threads--;
        };

        // Start threads
        std::vector<std::thread> threads_list;
        for (int i = 0; i < threads; i++)
            threads_list.push_back(std::thread(thread_loop, i));

        // Write games to file
        while (n_active_threads > 0 || !queue.empty())
        {
            queue.pop().write(output);
            n_games_completed++;
            int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            int games_per_hour = 3600000 * n_games_completed / (elapsed + 1);
            std::cout << "\r"
                      << n_games_completed << " games completed in "
                      << elapsed / 1000 << " s ("
                      << games_per_hour << " games/h)"
                      << std::flush;

            // Have we reached the maximum number of games?
            if (max_num_games > 0 && n_games_completed >= max_num_games)
            {
                stop.store(true, std::memory_order_relaxed);
                break;
            }
        }
        
        // Wait for threads to finish
        for (std::thread& t : threads_list)
            t.join();
        std::cout << std::endl;

        // Restore options
        UCI::Options::ShallowDepthPruning = sdp;
        UCI::Options::UCI_Chess960 = init_chess960;
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
                    ofile << board.to_fen() << " " << board.to_uci(node.move) << " " << node.score << "\n";
                    // Make the move
                    assert(board.legal(node.move) && "Illegal move!");
                    board = board.make_move(node.move);
                }
        }
    }


    bool file_valid(std::string filename, bool stats)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Failed to open file" << std::endl;
            return false;
        }

        // Fetch eof
        file.seekg(0, std::ios_base::end);
        auto eof = file.tellg();
        file.seekg(0, std::ios_base::beg);

        BinaryGame game;
        std::size_t game_number = 0;
        std::size_t move_number_total = 0;
        int64_t score_sum = 0;
        int64_t init_score_sum = 0;
        std::size_t abs_score_sum = 0;
        std::size_t abs_init_score_sum = 0;
        std::size_t draws = 0;
        std::size_t results = 0;
        std::size_t king_squares[NUM_SQUARES] = {0};
        while(file.tellg() < eof)
        {
            game_number++;
            std::size_t move_number = 0;

            if (!BinaryGame::read(file, game))
            {
                std::cerr << "Bad game " << game_number << std::endl;
                return false;
            }

            // Stats
            init_score_sum += game.nodes.front().score;
            abs_init_score_sum += abs(game.nodes.front().score);
            results += game.nodes.back().score + 1;
            if (game.nodes.back().score == 0)
                draws++;

            // Invalid initial position
            Board board(game.starting_pos);
            if (!board.valid())
            {
                std::cerr << "Game " << game_number << ": Invalid initial position " << std::endl;
                return false;
            }

            // Loop over all moves ensuring they are legal
            for (BinaryNode node : game.nodes)
            {
                move_number++;
                if (node.move != MOVE_NULL)
                {
                    score_sum += node.score;
                    abs_score_sum += abs(node.score);
                    king_squares[board.get_pieces(WHITE, KING).bitscan_forward()]++;
                    king_squares[vertical_mirror(board.get_pieces(BLACK, KING).bitscan_forward())]++;
                    if(!board.legal(node.move))
                    {
                        std::cerr << "Game " << game_number << " (move " << move_number << "): "
                                  << "Illegal move " << board.to_uci(node.move)
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
            move_number_total += move_number;
        }

        if (stats)
        {
            std::cout << filename << std::endl;
            std::cout << "  - Games: " << game_number << std::endl;
            std::cout << "  - Avg Moves: " << move_number_total / game_number << std::endl;
            std::cout << "  - Avg Score: " << score_sum / int64_t(move_number_total) << std::endl;
            std::cout << "  - Avg Init Score: " << init_score_sum / int64_t(game_number) << std::endl;
            std::cout << "  - Avg Score (abs): " << abs_score_sum / move_number_total << std::endl;
            std::cout << "  - Avg Init Score (abs): " << abs_init_score_sum / game_number << std::endl;
            std::cout << "  - Draw Ratio (%): " << 100.0 * draws / game_number << std::endl;
            std::cout << "  - Expected score: " << 0.5 * results / game_number << std::endl;
            std::cout << "  - King squares" << std::endl;
            std::cout << "  +---------+---------+---------+---------+---------+---------+---------+---------+" << std::endl;
            char file_labels[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'};
            char rank_labels[] = {'1', '2', '3', '4', '5', '6', '7', '8'};
            for (int rank = 7; rank >= 0; rank--)
            {
                for (int file = 0; file < 8; file++)
                    std::cout << "  |    "
                              << file_labels[file] << rank_labels[rank] << " ";
                std::cout << "  |  " << std::endl;
                for (int file = 0; file < 8; file++)
                    std::cout << "  |  "
                              << std::setw(5)
                              << std::fixed
                              << std::setprecision(2)
                              << 50.0 * king_squares[make_square(rank, file)] / move_number_total;
                std::cout << "  |"  << std::endl
                          << "  |    %    |    %    |    %    |    %    |    %    |    %    |    %    |    %    |" << std::endl
                          << "  +---------+---------+---------+---------+---------+---------+---------+---------+" << std::endl;
            }
        }

        return true;
    }


    void check_games(std::istringstream& stream)
    {
        // Files to parse
        bool stats = false;
        std::string token;
        std::vector<std::string> files;
        while (stream >> token)
        {
            if (token == "--stats")
                stats = true;
            else
                files.push_back(token);
        }
        assert(files.size() > 0 && "No input files have been passed!");

        // Store Chess960 status
        bool init_chess960 = UCI::Options::UCI_Chess960;
        UCI::Options::UCI_Chess960 = true;

        // Parse each file
        for (auto filename : files)
            std::cout << (file_valid(filename, stats) ? " [ OK ] " : " [FAIL] ")
                      << filename
                      << std::endl;
        
        // Restore Chess960 status
        UCI::Options::UCI_Chess960 = init_chess960;
    }


    void repair_games(std::istringstream& stream)
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

        // Fetch eof
        ifile.seekg(0, std::ios_base::end);
        auto eof = ifile.tellg();
        ifile.seekg(0, std::ios_base::beg);

        // Store Chess960 status
        bool init_chess960 = UCI::Options::UCI_Chess960;
        UCI::Options::UCI_Chess960 = true;

        // Read entire file
        BinaryGame game;
        std::size_t bad_games = 0;
        std::size_t game_number = 0;
        while(ifile.tellg() < eof)
        {
            // Read game
            game_number++;
            if (!BinaryGame::read(ifile, game))
            {
                bad_games++;
                std::cout << "Bad game " << game_number << std::endl;
                continue;
            }

            // Invalid initial position
            Board board(game.starting_pos);
            if (!board.valid())
            {
                bad_games++;
                std::cerr << "Invalid initial position in game " << game_number << std::endl;
                continue;
            }

            // Loop over all moves ensuring they are legal
            bool good_game = true;
            Score last_score = game.nodes.back().score;
            for (BinaryNode& node : game.nodes)
            {
                if (node.move != MOVE_NULL)
                {
                    if(!board.legal(node.move))
                    {
                        good_game = false;
                        std::cout << "Illegal move in game " << game_number << std::endl;
                        break;
                    }
                    board = board.make_move(node.move);
                    last_score = node.score;
                }
                // Sanity checks for the end node
                else
                {
                    // Ensure the last score is the game outcome (-1, 0 or 1)
                    if (abs(node.score) > 1)
                    {
                        good_game = false;
                        std::cout << "Bad termination in game " << game_number << std::endl;
                        break;
                    }

                    // Check if the last score matches the game outcome
                    if ((last_score == 0 && node.score != 0) ||
                        (last_score > 0 && node.score != WHITE_COLOR) ||
                        (last_score < 0 && node.score != BLACK_COLOR))
                    {
                        std::cout << "Fixing bad game termination in game " << game_number
                                  << " with score " << last_score
                                  << " and result " << node.score
                                  << std::endl;
                        node.score = last_score > 0 ? WHITE_COLOR
                                   : last_score < 0 ? BLACK_COLOR
                                   : NO_COLOR;
                    }
                }
            }

            // Write the game if it is good
            if (good_game)
                game.write(ofile);
            else
                bad_games++;
        }
        std::cout << "Written " << game_number - bad_games
                  << " games to " << output_file_path
                  << " and discarded " << bad_games
                  << " bad games" << std::endl;
        ofile.close();    
    
        // Restore Chess960 status
        UCI::Options::UCI_Chess960 = init_chess960;
    }


    void rescore_games(std::istringstream& stream)
    {
        // Parameters and their default values
        std::string input_file_path;
        std::string output_file_path;
        int depth = 8;
        int nodes = 0;
        int threads = 1;
        int hash = 1;
        bool shallow_depth_pruning = false;
        std::string syzygy_path = "";
        bool chess_960 = false;

        // Parameter reading
        assert((stream >> input_file_path) && "Input file path required!");
        assert((stream >> output_file_path) && "Output file path required!");

        // Read passed parameters
        std::string token;
        while (stream >> token)
            if (token == "depth")
                stream >> depth;
            else if (token == "nodes")
                stream >> nodes;
            else if (token == "threads")
                stream >> threads;
            else if (token == "hash")
                stream >> hash;
            else if (token == "shallow_depth_pruning")
                shallow_depth_pruning = true;
            else if (token == "syzygy_path")
                stream >> syzygy_path;
            else if (token == "chess_960")
                chess_960 = true;
            else
            {
                std::cout << "Unknown option " << token << std::endl;
                return;
            }

        // Load Syzygy tablebases
        if (syzygy_path != "")
            Syzygy::load(syzygy_path);

        // Update Chess960 status
        bool init_chess960 = UCI::Options::UCI_Chess960;
        UCI::Options::UCI_Chess960 = chess_960;

        // Open files
        std::ifstream ifile(input_file_path);
        std::ofstream ofile(output_file_path);
        assert(ifile.is_open() && "Failed to open input file!");
        assert(ofile.is_open() && "Failed to open output file!");

        // Current number of games
        std::size_t n_games_completed = 0;

        // Prepare search data
        Search::Limits limits;
        if (depth > 0)
            limits.depth = depth;
        if (nodes > 0)
            limits.nodes = nodes;
        std::atomic_bool stop = false;
        ThreadSafeQueueWithCapacity<BinaryGame> input_queue(2 * threads);
        ThreadSafeQueue<BinaryGame> output_queue;
        std::atomic_int n_active_threads = threads;
        auto start = std::chrono::steady_clock::now();

        // Store and reset shallow depth pruning
        bool sdp = UCI::Options::ShallowDepthPruning;
        UCI::Options::ShallowDepthPruning = shallow_depth_pruning;

        auto thread_scorer = [=, &input_queue, &output_queue, &n_active_threads, &stop]()
        {
            // Prepare search thread
            ThreadPool pool(1, hash);
            Thread& thread = pool.front();

            // Data generation loop
            while (!stop.load(std::memory_order_relaxed))
            {
                // Pick the next game
                BinaryGame game;
                if (!input_queue.pop(game))
                    break;

                // Game loop
                pool.clear();
                Position pos(game.starting_pos.fen());
                for (BinaryNode& node : game.nodes)
                {
                    if (node.move == MOVE_NULL)
                        break;

                    pos.set_init_ply();
                    SearchResult result = thread.simple_search(pos, limits, threads > 1);
                    node.score = result.score;
                    pos.make_move(node.move);
                }
                output_queue.push(game);
            }
            n_active_threads--;
        };

        auto thread_reader = [=, &ifile, &input_queue, &stop]()
        {
            // Fetch eof
            ifile.seekg(0, std::ios_base::end);
            auto eof = ifile.tellg();
            ifile.seekg(0, std::ios_base::beg);

            // Read entire file
            BinaryGame game;
            std::string line;
            while(ifile.tellg() < eof)
            {
                // Read game: starting FEN
                if (!std::getline(ifile, line))
                    break;
                
                // Initialise position
                Position pos(line);
                game.begin(pos.board());

                // Read moves until the termination node
                bool valid_game = true;
                while (std::getline(ifile, line))
                {
                    // Check if the line does not start with "game:"
                    if (line.rfind("game:", 0) == 0)
                        break;

                    // Parse move
                    Move move = UCI::move_from_uci(pos, line);
                    if (!pos.board().legal(move))
                    {
                        std::cerr << "Illegal move " << pos.board().to_uci(move)
                                  << " in position " << pos.board().to_fen()
                                  << std::endl;
                        valid_game = false;
                    }
                    if (valid_game)
                    {
                        pos.make_move(move);
                        pos.set_init_ply();
                        game.push(move, SCORE_NONE);
                    }
                }

                // Read result, add termination node and push to the queue
                if (!valid_game)
                    continue;
                game.push(MOVE_NULL, std::stoi(line.substr(6)));
                input_queue.push(game);
            }
            input_queue.close();
        };

        // Start reader thread
        std::thread reader_thread(thread_reader);

        // Start threads
        std::vector<std::thread> threads_list;
        for (int i = 0; i < threads; i++)
            threads_list.push_back(std::thread(thread_scorer));

        // Write games to file
        while (n_active_threads > 0 || !output_queue.empty())
        {
            // Pop last game from the threads
            output_queue.pop().write(ofile);
            n_games_completed++;
            int64_t elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            int games_per_hour = 3600000 * n_games_completed / (elapsed + 1);
            std::cout << "\r"
                      << n_games_completed << " games completed in "
                      << elapsed / 1000 << " s ("
                      << games_per_hour << " games/h)"
                      << std::flush;
        }
        
        // Wait for threads to finish
        reader_thread.join();
        for (std::thread& t : threads_list)
            t.join();
        std::cout << std::endl;

        // Restore options
        UCI::Options::ShallowDepthPruning = sdp;
        UCI::Options::UCI_Chess960 = init_chess960;
    }
}
