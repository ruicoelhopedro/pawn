
#include "Types.hpp"
#include "Position.hpp"
#include "Thread.hpp"
#include "UCI.hpp"
#include "data_gen/data_gen.hpp"
#include "data_gen/data_loader.hpp"
#include <fstream>
#include <string>
#include <vector>

extern "C"
{
    void init_pawn()
    {
        Bitboards::init_bitboards();
        Zobrist::build_rnd_hashes();
        NNUE::init();
        UCI::init();
        UCI::Options::UCI_Chess960 = true;
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



    Size get_nnue_data(
        const char* file_name,
        const Size* indices,
        const Size* selection,
        Size n_selected,
        Size seed,
        Size prob,
        Size* w_idx,
        Size* b_idx,
        unsigned short* w_cols,
        unsigned short* b_cols,
        short* scores,
        char* results,
        char* buckets
    )
    {
        // Open files
        std::string fname_str(file_name);
        std::ifstream ifile(fname_str, std::ios::binary);
        assert(ifile.is_open() && "Failed to open input file!");

        PseudoRandom prng(seed);

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

            int bucket = 4 * rank(ks) + file(ks);

            return s
                 + p      * NUM_SQUARES
                 + t      * NUM_SQUARES * NUM_PIECE_TYPES
                 + bucket * NUM_SQUARES * NUM_PIECE_TYPES * NUM_COLORS;
        };

        Size count = 0;
        BinaryGame game;
        for (Size i = 0; i < n_selected; i++)
        {
            // Read the game at the specified index
            ifile.seekg(indices[selection[i]]);
            std::size_t game_size = indices[selection[i] + 1] - indices[selection[i]];
            if (!BinaryGame::read(ifile, game, game_size))
                continue;

            // Find winner
            Color result = Color(game.nodes.back().score);
    
            // Write each position, score and move
            std::size_t num_pos = 0; 
            Board board(game.starting_pos, true);
            for (BinaryNode node : game.nodes)
                if (node.move != MOVE_NULL)
                {
                    // Write to files if the position verifies
                    // i) Not in check
                    // ii) Best move not a capture
                    // iii) Best move not a promotion
                    if (!board.checkers() &&
                        !node.move.is_capture() &&
                        (prng.next(prob) < 1 || num_pos == 0))
                    {
                        // Update features
                        w_idx[count+1] = w_idx[count];
                        b_idx[count+1] = b_idx[count];

                        // Set the pointers based on the side to move
                        unsigned short* w_cols_ptr = board.turn() == WHITE ? w_cols : b_cols;
                        unsigned short* b_cols_ptr = board.turn() == WHITE ? b_cols : w_cols;
                        Size* w_idx_ptr = board.turn() == WHITE ? w_idx : b_idx;
                        Size* b_idx_ptr = board.turn() == WHITE ? b_idx : w_idx;

                        // Get the kings and pieces
                        Square king[] = {
                            board.get_pieces<WHITE, KING>().bitscan_forward(),
                            vertical_mirror(board.get_pieces<BLACK, KING>().bitscan_forward())
                        };
                        Bitboard pieces[] = {
                            board.get_pieces<WHITE>(),
                            board.get_pieces<BLACK>()
                        };

                        for (Square s = 0; s < NUM_SQUARES; s++)
                        {
                            Square vm = vertical_mirror(s);
                            PieceType piece = board.get_piece_at(s);
                            if (piece != PIECE_NONE)
                            {
                                if (pieces[WHITE].test(s))
                                {
                                    w_cols_ptr[w_idx_ptr[count+1]++] = index(WHITE, piece, s, king[WHITE]);
                                    b_cols_ptr[b_idx_ptr[count+1]++] = index(BLACK, piece, vm, king[BLACK]);
                                }
                                else
                                {
                                    w_cols_ptr[w_idx_ptr[count+1]++] = index(BLACK, piece, s, king[WHITE]);
                                    b_cols_ptr[b_idx_ptr[count+1]++] = index(WHITE, piece, vm, king[BLACK]);
                                }
                            }
                        }

                        // Store eval and phase
                        scores[count] = node.score * turn_to_color(board.turn());
                        buckets[count] = (board.get_pieces().count() - 1) / 8;
                        results[count] = result * turn_to_color(board.turn());
                        count++;
                        num_pos++;
                    }

                    // Make the move
                    board = board.make_move(node.move);
                }
        }

        return count;
    }
}
