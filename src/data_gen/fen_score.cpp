#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/fen_score.hpp"
#include <string>
#include <variant>


namespace FEN_Scores
{

    FENFileFormat::FENFileFormat(std::string prefix)
        : file(prefix + "evals.dat")
    {
        assert(file.is_open() && "Failed to open output file!");
    }


    FENFileFormat::FENFileFormat(int depth, std::string prefix)
        : file(prefix + "scores-depth" + std::to_string(depth) + ".dat")
    {
        assert(file.is_open() && "Failed to open output file!");
    }


    void FENFileFormat::write(const Position& pos, const std::string& fen,  Score s)
    {
        file << fen << " " << s << "\n";
    }


    void FENFileFormat::write(const Position& pos, const std::string& fen, Move m, Score s)
    {
        file << fen << " " << m << " " << s << "\n";
    }


    BinaryFileFormat::BinaryFileFormat(std::string prefix)
        : file(prefix + "evals.dat")
    {
        assert(file.is_open() && "Failed to open output file!");
    }


    BinaryFileFormat::BinaryFileFormat(int depth, std::string prefix)
        : file(prefix + "scores-depth" + std::to_string(depth) + ".dat")
    {
        assert(file.is_open() && "Failed to open output file!");
    }


    void BinaryFileFormat::write(const Position& pos, const std::string& fen, Score s)
    {
        BinaryBoard bb(pos.board());
        file.write(reinterpret_cast<const char*>(&bb), sizeof(BinaryBoard));
        file.write(reinterpret_cast<const char*>(&s), sizeof(Score));
    }


    void BinaryFileFormat::write(const Position& pos, const std::string& fen, Move m, Score s)
    {
        BinaryBoard bb(pos.board());
        file.write(reinterpret_cast<const char*>(&bb), sizeof(BinaryBoard));
        file.write(reinterpret_cast<const char*>(&m), sizeof(Move));
        file.write(reinterpret_cast<const char*>(&s), sizeof(Score));
    }


    void score_fens(std::istringstream& stream)
    {
        // Generation parameters
        std::string file_name;
        int depth;
        std::string option;
        std::string prefix;

        // Required parameters
        assert((stream >> file_name) && "Failed to parse book file name!");
        assert((stream >> depth) && "Failed to parse search depth!");

        // Optional parameters
        if (!(stream >> option)) option = "fen";
        if (!(stream >> prefix)) prefix = "";

        // Build output file format
        std::variant<FENFileFormat, BinaryFileFormat> output_var = FENFileFormat(depth, prefix);
        if (option == "binary")
            output_var = BinaryFileFormat(depth, prefix);

        // Read book
        auto fens = read_fens(file_name);

        // Initialise limits
        Search::Limits limits;
        limits.depth = depth;

        // Disable shallow depth pruning
        bool sdp = UCI::Options::ShallowDepthPruning;
        UCI::Options::ShallowDepthPruning = false;

        // Loop over book
        for (auto fen : fens)
        {
            // Clear data
            UCI::ucinewgame(stream);
            
            // Search the position
            Position pos(fen);
            SearchResult result = pool->front().simple_search(pos, limits);

            // Parse output
            Move m = result.bestmove;
            Score s = 100 * result.score / PawnValue.endgame();

            // Write to the output file
            std::visit([pos, fen, m, s] (auto& output) { output.write(pos, fen, m, s); },
                       output_var);
        }

        // Restore shallow depth pruning
        UCI::Options::ShallowDepthPruning = sdp;
    }


    void evaluate_fens(std::istringstream& stream)
    {
        // Generation parameters
        std::string file_name;
        std::string option;
        std::string prefix;

        // Required parameters
        assert((stream >> file_name) && "Failed to parse book file name!");

        // Optional parameters
        if (!(stream >> option)) option = "fen";
        if (!(stream >> prefix)) prefix = "";

        // Build output file format
        std::variant<FENFileFormat, BinaryFileFormat> output_var = FENFileFormat(prefix);
        if (option == "binary")
            output_var = BinaryFileFormat(prefix);

        // Read book
        auto fens = read_fens(file_name);

        // Loop over book
        for (auto fen : fens)
        {            
            // Evaluate the position
            Position pos(fen);
            Score s = 100 * Evaluation::evaluation(pos.board()) / PawnValue.endgame();

            // Write to the output file
            std::visit([pos, fen, s] (auto& output) { output.write(pos, fen, s); },
                       output_var);
        }
    }
}