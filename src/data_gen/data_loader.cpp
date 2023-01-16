
#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/data_loader.hpp"
#include "data_gen/psqt.hpp"
#include <fstream>
#include <string>
#include <vector>

extern "C"
{
    void init_pawn()
    {
        Bitboards::init_bitboards();
        Zobrist::build_rnd_hashes();
        PSQT::init();
        UCI::init_options();
        pool = new ThreadPool();
    }



    Size get_num_games(const char* file_name)
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str);
        assert(ifile.is_open() && "Failed to open input file!");

        Size count = 0;

        // Read entire file
        BinaryGame game;
        while(BinaryGame::read(ifile, game))
            count++;

        return count;
    }



    void get_indices(const char* file_name, Size* indices)
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str, std::ios::binary);
        assert(ifile.is_open() && "Failed to open input file!");

        Size count = 0;
        indices[count++] = 0;

        // Read entire file
        BinaryGame game;
        while(BinaryGame::read(ifile, game))
            indices[count++] = ifile.tellg();
    }



    Size get_num_positions(const Size* indices, const Size* selection, Size n_selected)
    {
        constexpr std::size_t BoardSize = sizeof(BinaryBoard);
        constexpr std::size_t MoveSize = sizeof(BinaryNode);
        constexpr std::size_t ExcessSize = BoardSize + MoveSize;
        
        Size count = 0;
        for (Size i = 0; i < n_selected; i++)
            count += (indices[selection[i]+1] - indices[selection[i]] - ExcessSize) / MoveSize;

        return count;
    }



    Size get_dense_psq_data(const char* file_name, const Size* selection, Size n_selected, char* X, short* evals, char* results)
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str, std::ios::binary);
        assert(ifile.is_open() && "Failed to open input file!");

        constexpr std::size_t NumFeatures = PSQT_DataGen::Mapper::N_FEATURES;

        Size count = 0;
        BinaryGame game;
        for (Size i = 0; i < n_selected; i++)
        {
            // Read the game at the specified index
            ifile.seekg(selection[i]);
            BinaryGame::read(ifile, game);

            // Find winner
            Color result = Color(game.nodes.back().score);
    
             // Write each position, score and move
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                    {
                        // Write to files if the position verifies
                        // i) Not in check
                        // ii) Best move not a capture
                        // iii) Best move not a promotion
                        if (!board.checkers() &&
                            !node.move.is_capture() &&
                            !node.move.is_promotion())
                        {
                            PSQT_DataGen::FeatureSample sample(board, node.score, result);
                            // Update features
                            for (uint f = 0; f < sample.count; f++)
                                X[count * NumFeatures + sample.features[f]] = sample.color[f];
                            // Store eval and result
                            evals[count] = sample.eval;
                            results[count] = sample.result;
                            count++;
                        }

                        // Make the move
                        board = board.make_move(node.move);
                    }
        }

        return count;
    }


    Size get_sparse_psq_data(const char* file_name, const Size* selection, Size n_selected, Size* idx, unsigned int* cols, char* colors, short* evals, char* results)
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str, std::ios::binary);
        assert(ifile.is_open() && "Failed to open input file!");

        idx[0] = 0;

        Size count = 0;
        BinaryGame game;
        for (Size i = 0; i < n_selected; i++)
        {
            // Read the game at the specified index
            ifile.seekg(selection[i]);
            BinaryGame::read(ifile, game);

            // Find winner
            Color result = Color(game.nodes.back().score);
    
             // Write each position, score and move
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                    {
                        // Write to files if the position verifies
                        // i) Not in check
                        // ii) Best move not a capture
                        // iii) Best move not a promotion
                        if (!board.checkers() &&
                            !node.move.is_capture() &&
                            !node.move.is_promotion())
                        {
                            PSQT_DataGen::FeatureSample sample(board, node.score, result);
                            // Update features
                            idx[count+1] = idx[count];
                            for (uint f = 0; f < sample.count; f++)
                            {
                                cols[idx[count+1]] = sample.features[f];
                                colors[idx[count+1]] = sample.color[f];
                                idx[count+1] += 1;
                            }
                            // Store eval and result
                            evals[count] = sample.eval;
                            results[count] = sample.result;
                            count++;
                        }

                        // Make the move
                        board = board.make_move(node.move);
                    }
        }

        return count;
    }



    Size get_positional_data(const char* file_name, const Size* selection, Size n_selected, Size* w_idx, Size* b_idx, unsigned short* w_cols, unsigned short* b_cols, short* evals, char* phases)
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str, std::ios::binary);
        assert(ifile.is_open() && "Failed to open input file!");

        w_idx[0] = 0;
        b_idx[0] = 0;

        auto index = [](Turn t, PieceType p, Square s, Square ks)
        {
            // Mirror if king on the files E to H
            if (file(ks) >= 4)
            {
                s = horizontal_mirror(s);
                ks = horizontal_mirror(ks);
            }

            int ki = 4 * rank(ks) + file(ks);

            int index = s
                      + p  * 64
                      + ki * 64 * 5
                      + t  * 64 * 5 * 32;

            return index;
        };

        Size count = 0;
        BinaryGame game;
        for (Size i = 0; i < n_selected; i++)
        {
            // Read the game at the specified index
            ifile.seekg(selection[i]);
            BinaryGame::read(ifile, game);
    
             // Write each position, score and move
            Board board(game.starting_pos);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                    {
                        // Write to files if the position verifies
                        // i) Not in check
                        // ii) Best move not a capture
                        // iii) Best move not a promotion
                        if (!board.checkers() &&
                            !node.move.is_capture() &&
                            !node.move.is_promotion())
                        {
                            // Update features
                            w_idx[count+1] = w_idx[count];
                            b_idx[count+1] = b_idx[count];

                            Square king[] = { board.get_pieces<WHITE, KING>().bitscan_forward(),
                                              board.get_pieces<BLACK, KING>().bitscan_forward() };
                            Bitboard pieces[] = { board.get_pieces<WHITE>(),
                                                  board.get_pieces<BLACK>() };

                            for (Square s = 0; s < NUM_SQUARES; s++)
                            {
                                PieceType piece = board.get_piece_at(s);
                                if (piece != PIECE_NONE && piece != KING)
                                {
                                    if (pieces[WHITE].test(s))
                                    {
                                        w_cols[w_idx[count+1]++] = index(WHITE, piece, s, king[WHITE]);
                                        b_cols[b_idx[count+1]++] = index(BLACK, piece, vertical_mirror(s), vertical_mirror(king[BLACK]));
                                    }
                                    else
                                    {
                                        w_cols[w_idx[count+1]++] = index(BLACK, piece, s, king[WHITE]);
                                        b_cols[b_idx[count+1]++] = index(WHITE, piece, vertical_mirror(s), vertical_mirror(king[BLACK]));
                                    }
                                }
                            }

                            // Store eval and phase
                            evals[count] = node.score - pool->front().evaluate<false>(board) + board.psq().tapered(board.phase());
                            phases[count] = MixedScore(64, 0).tapered(board.phase());
                            count++;
                        }

                        // Make the move
                        board = board.make_move(node.move);
                    }
        }

        return count;
    }
}