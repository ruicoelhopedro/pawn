#include "data_gen/data_gen.hpp"
#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/psqt.hpp"
#include <iostream>
#include <fstream>



namespace PSQT_DataGen
{
    FileFormat::FileFormat(int offset, std::string prefix)
    {
        std::string suffix = "+" + std::to_string(offset) + ".dat";

        rows    = std::ofstream(prefix + "rows"    + suffix);
        evals   = std::ofstream(prefix + "evals"   + suffix);
        colors  = std::ofstream(prefix + "colors"  + suffix);
        indices = std::ofstream(prefix + "indices" + suffix);
        results = std::ofstream(prefix + "results" + suffix);

        assert(   rows.is_open() && "Failed to open rows file!");
        assert(  evals.is_open() && "Failed to open evals file!");
        assert( colors.is_open() && "Failed to open colors file!");
        assert(indices.is_open() && "Failed to open indices file!");
        assert(results.is_open() && "Failed to open results file!");
    }



    FeatureSample::FeatureSample(const Board& board, Eval eval_input, Color result_input)
    {
        // Helpers
        Square king[] = { board.get_pieces<WHITE, KING>().bitscan_forward(),
                        board.get_pieces<BLACK, KING>().bitscan_forward() };
        Bitboard pieces[] = { board.get_pieces<WHITE>(),
                            board.get_pieces<BLACK>() };

        // Populate features
        count = 0;
        for (Square s = 0; s < NUM_SQUARES; s++)
        {
            PieceType piece = board.get_piece_at(s);
            if (piece != PIECE_NONE && piece != KING)
            {
                if (pieces[WHITE].test(s))
                {
                    push(Mapper::map(WHITE, piece, s, king[WHITE]), WHITE_COLOR);
                    push(Mapper::map(BLACK, piece, s, king[BLACK]), WHITE_COLOR);
                }
                else
                {
                    push(Mapper::map(WHITE, piece, vertical_mirror(s), vertical_mirror(king[BLACK])), BLACK_COLOR);
                    push(Mapper::map(BLACK, piece, vertical_mirror(s), vertical_mirror(king[WHITE])), BLACK_COLOR);
                }
            }
        }

        // Compute eval difference
        Score static_eval = pool->front().evaluate<false>(board);
        Score psq = board.psq().tapered(board.phase());
        eval = eval_input - (static_eval - psq);

        // Store winner
        result = Result(result_input);
    }


    void FeatureSample::write(FileFormat& file)
    {
        file.indices.write(reinterpret_cast<const char*>(&count), sizeof(FeatureCount));
        file.rows.write(reinterpret_cast<const char*>(&features), count * sizeof(Feature));
        file.colors.write(reinterpret_cast<const char*>(&color), count * sizeof(PieceColor));
        file.evals.write(reinterpret_cast<const char*>(&eval), sizeof(Eval));
        file.results.write(reinterpret_cast<const char*>(&result), sizeof(Result));
    }


    void FeatureSample::push(Feature feature, Color pc)
    {
        features[count] = feature;
        color[count] = pc;
        count++;
    }



    std::size_t Mapper::map(Turn turn, PieceType piece, Square square, Square king_sq)
    {
        // Mirror if king on the files E to H
        if (file(king_sq) >= 4)
        {
            square = horizontal_mirror(square);
            king_sq = horizontal_mirror(king_sq);
        }

        int king_index = 4 * rank(king_sq) + file(king_sq);

        int index = square
                + piece      *  FEATURE_DIMS[0]
                + king_index * (FEATURE_DIMS[0] * FEATURE_DIMS[1])
                + turn       * (FEATURE_DIMS[0] * FEATURE_DIMS[1] * FEATURE_DIMS[2]);

        return index;
    }


    Turn Mapper::turn(std::size_t index)
    {
        return Turn(index / (FEATURE_DIMS[0] * FEATURE_DIMS[1] * FEATURE_DIMS[2]));
    }
    Square Mapper::king_sq(std::size_t index)
    {
        return Square((index / (FEATURE_DIMS[0] * FEATURE_DIMS[1])) % FEATURE_DIMS[2]);
    }
    PieceType Mapper::piece(std::size_t index)
    {
        return PieceType((index / FEATURE_DIMS[0]) % FEATURE_DIMS[1]);
    }
    Square Mapper::square(std::size_t index)
    {
        return Square(index % FEATURE_DIMS[0]);
    }



    void gen_data_psqt(std::istringstream& stream)
    {
        // Generation parameters
        std::string file_name;
        int depth;
        int offset;
        int interval;
        std::string prefix;

        // Required parameters
        assert((stream >> file_name) && "Failed to parse book file name!");
        assert((stream >> depth) && "Failed to parse search depth!");

        // Optional parameters
        if (!(stream >> interval)) interval = 1;
        if (!(stream >> offset))   offset = 0;
        if (!(stream >> prefix))   prefix = "";

        // Build output file format
        FileFormat output(offset, prefix);

        // Read book
        auto fens = read_fens(file_name);

        // Loop over book
        for (std::size_t i = offset; i < fens.size(); i += interval)
        {
            std::cout << "\r" << (i + 1) << "/" << fens.size() << std::flush;

            // Clear data
            UCI::ucinewgame(stream);
            
            // Play the game
            GameResult result = pool->front().play_game(fens[i], depth, 1500);

            // Write to files if the position verifies
            // i) Not in check
            // ii) Best move not a capture
            // iii) Best move not a promotion
            for (auto node : result.game)
                if (!node.board.checkers() &&
                    !node.bestmove.is_capture() &&
                    !node.bestmove.is_promotion())
                    FeatureSample(node.board, node.score, result.result).write(output);
        }
        std::cout << std::endl;
    }
}