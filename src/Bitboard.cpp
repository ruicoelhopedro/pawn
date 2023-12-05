#include "Bitboard.hpp"
#include "Types.hpp"
#include <algorithm>


// IO operators
std::ostream& operator<<(std::ostream& out, const Bitboard& bb)
{
    for (int i = 7; i >= 0; i--)
    {
        for (int j = 0; j < 8; j++)
        {
            int index = i * 8 + j;
            if (bb.test(index))
                out << " x";
            else
                out << " .";
        }
        out << "\n";
    }
    return out;
}

constexpr int Bitboard::index64[64];

namespace Bitboards
{
    // Global variables
    Bitboard diagonals[NUM_SQUARES];
    Bitboard ranks_files[NUM_SQUARES];
    Bitboard pseudo_attacks[NUM_PIECE_TYPES][NUM_SQUARES];
    Bitboard pawn_attacks[NUM_COLORS][NUM_SQUARES];
    Bitboard castle_non_attacked_squares[NUM_COLORS][NUM_CASTLE_SIDES];
    Bitboard castle_non_occupied_squares[NUM_COLORS][NUM_CASTLE_SIDES];
    Square castle_target_square[NUM_COLORS][NUM_CASTLE_SIDES];
    MagicBitboard bishop_magics[NUM_SQUARES];
    MagicBitboard rook_magics[NUM_SQUARES];
    Bitboard between_squares[NUM_SQUARES][NUM_SQUARES];

    void init_bitboards()
    {
        build_diagonals();
        build_ranks_files();

        for (int square = 0; square < NUM_SQUARES; square++)
            build_pseudo_attacks(static_cast<Square>(square));

        build_castle_squares();
        magic_helpers::gen_all_magics(false);
        build_between_bbs();
    }

    Bitboard isolated_mask(Bitboard open_files)
    {
        constexpr Direction Left = -1;
        constexpr Direction Right = 1;
        Bitboard l_bb = (open_files & a_file).shift< Left>() | h_file;
        Bitboard r_bb = (open_files & h_file).shift<Right>() | a_file;
        return l_bb & r_bb;
    }
    
    int file_count(Bitboard file_bb)
    {
        return (file_bb & rank_1).count();
    }

    Bitboard between(Square s1, Square s2)
    {
        return between_squares[s1][s2];
    }

    void build_diagonals()
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                // Reset the bitboard
                diagonals[j + 8 * i] = Bitboards::empty;

                for (int k = -8; k <= 8; k++)
                {
                    int ti = i + k;
                    // Major
                    int tj = j + k;
                    if (inside_board(ti, tj))
                        diagonals[j + 8 * i].set(ti * 8 + tj);
                    // Minor
                    tj = j - k;
                    if (inside_board(ti, tj))
                        diagonals[j + 8 * i].set(ti * 8 + tj);
                }
            }
        }
    }

    void build_ranks_files()
    {
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                ranks_files[j + 8 * i] = ranks[i] | files[j];
    }

    void build_pseudo_attacks(Square square)
    {
        // Clear everything
        for (int piece = 0; piece < NUM_PIECE_TYPES; piece++)
            pseudo_attacks[piece][square] = empty;

        pawn_attacks[WHITE][square] = pseudo_attacks_pawns<WHITE>(square);
        pawn_attacks[BLACK][square] = pseudo_attacks_pawns<BLACK>(square);
        pseudo_attacks[KNIGHT][square] = pseudo_attacks_knights(square);
        pseudo_attacks[BISHOP][square] = pseudo_attacks_bishops(square);
        pseudo_attacks[ROOK][square] = pseudo_attacks_rooks(square);
        pseudo_attacks[QUEEN][square] = pseudo_attacks_queens(square);
        pseudo_attacks[KING][square] = pseudo_attacks_kings(square);
    }

    Bitboard pseudo_attacks_knights(Square square)
    {
        Bitboard result = empty;
        int di[] = { -1, 1, -2, 2, -2, 2, -1, 1 };
        int dj[] = { -2, -2, -1, -1, 1, 1, 2, 2 };
        for (int k = 0; k < 8; k++)
        {
            int x = rank(square) + di[k];
            int y = file(square) + dj[k];
            if (inside_board(x, y))
                result.set(make_square(x, y));
        }
        return result;
    }

    Bitboard pseudo_attacks_bishops(Square square)
    {
        auto result = diagonals[square];
        result.reset(square);
        return result;
    }

    Bitboard pseudo_attacks_rooks(Square square)
    {
        auto result = ranks_files[square];
        result.reset(square);
        return result;
    }

    Bitboard pseudo_attacks_queens(Square square)
    {
        return diagonals[square] ^ ranks_files[square];
    }

    Bitboard pseudo_attacks_kings(Square square)
    {
        Bitboard result = empty;
        int i = rank(square);
        int j = file(square);
        for (int di = std::max(0, i - 1); di <= std::min(7, i + 1); di++)
            for (int dj = std::max(0, j - 1); dj <= std::min(7, j + 1); dj++)
                if (i != di || j != dj)
                    result.set(make_square(di, dj));
        return result;
    }

    template<>
    Bitboard get_attacks<BISHOP>(Square square, Bitboard occupancy)
    {
        return bishop_magics[square].get_moveboard(occupancy);
    }

    template<>
    Bitboard get_attacks<ROOK>(Square square, Bitboard occupancy)
    {
        return rook_magics[square].get_moveboard(occupancy);
    }

    template<>
    Bitboard get_attacks<QUEEN>(Square square, Bitboard occupancy)
    {
        return bishop_magics[square].get_moveboard(occupancy) |
            rook_magics[square].get_moveboard(occupancy);
    }

    void build_castle_squares()
    {
        // Squares that cannot be attacked for castling to be legal
        castle_non_attacked_squares[WHITE][KINGSIDE].set(SQUARE_F1);
        castle_non_attacked_squares[WHITE][KINGSIDE].set(SQUARE_G1);
        castle_non_attacked_squares[WHITE][QUEENSIDE].set(SQUARE_D1);
        castle_non_attacked_squares[WHITE][QUEENSIDE].set(SQUARE_C1);
        castle_non_attacked_squares[BLACK][KINGSIDE].set(SQUARE_F8);
        castle_non_attacked_squares[BLACK][KINGSIDE].set(SQUARE_G8);
        castle_non_attacked_squares[BLACK][QUEENSIDE].set(SQUARE_D8);
        castle_non_attacked_squares[BLACK][QUEENSIDE].set(SQUARE_C8);

        // Free squares
        castle_non_occupied_squares[WHITE][KINGSIDE].set(SQUARE_F1);
        castle_non_occupied_squares[WHITE][KINGSIDE].set(SQUARE_G1);
        castle_non_occupied_squares[WHITE][QUEENSIDE].set(SQUARE_D1);
        castle_non_occupied_squares[WHITE][QUEENSIDE].set(SQUARE_C1);
        castle_non_occupied_squares[WHITE][QUEENSIDE].set(SQUARE_B1);
        castle_non_occupied_squares[BLACK][KINGSIDE].set(SQUARE_F8);
        castle_non_occupied_squares[BLACK][KINGSIDE].set(SQUARE_G8);
        castle_non_occupied_squares[BLACK][QUEENSIDE].set(SQUARE_D8);
        castle_non_occupied_squares[BLACK][QUEENSIDE].set(SQUARE_C8);
        castle_non_occupied_squares[BLACK][QUEENSIDE].set(SQUARE_B8);

        // Target squares
        castle_target_square[WHITE][KINGSIDE] = SQUARE_G1;
        castle_target_square[WHITE][QUEENSIDE] = SQUARE_C1;
        castle_target_square[BLACK][KINGSIDE] = SQUARE_G8;
        castle_target_square[BLACK][QUEENSIDE] = SQUARE_C8;
    }

    void build_between_bbs()
    {
        for (int i = 0; i < NUM_SQUARES; i++)
            for (int j = 0; j < NUM_SQUARES; j++)
                if (i == j)
                    between_squares[i][j] = Bitboard();
                else if (diagonals[i].test(j))
                    between_squares[i][j] = get_attacks<BISHOP>(static_cast<Square>(i), Bitboard::from_single_bit(j)) &
                                            get_attacks<BISHOP>(static_cast<Square>(j), Bitboard::from_single_bit(i));
                else if (ranks_files[i].test(j))
                    between_squares[i][j] = get_attacks<ROOK  >(static_cast<Square>(i), Bitboard::from_single_bit(j)) &
                                            get_attacks<ROOK  >(static_cast<Square>(j), Bitboard::from_single_bit(i));
    }





    MagicBitboard::MagicBitboard(uint64_t magic, Bitboard blockmask, const std::vector<Bitboard>& blockboards, const std::vector<Bitboard>& moveboards)
        : m_bits(blockmask.count()), m_magic(magic), m_blockmask(blockmask), m_moveboards(moveboards.size())
    {
        for (unsigned int i = 0; i < moveboards.size(); i++)
            m_moveboards[get_index(blockboards[i])] = moveboards[i];
    }
    int MagicBitboard::get_index(Bitboard blockboard) const
    {
        return (blockboard.to_uint64() * m_magic) >> (64 - m_bits);
    }
    Bitboard MagicBitboard::get_moveboard(Bitboard occupancy) const
    {
        Bitboard blockboard = occupancy & m_blockmask;
        return m_moveboards[get_index(blockboard)];
    }





    Bitboard magic_helpers::blockmask_rook(Square square)
    {
        Bitboard cross = pseudo_attacks_rooks(square);
        for (auto edge : { a_file, h_file, rank_1, rank_8 })
            if (!edge.test(square))
                cross &= (~edge);
        return cross;
    }

    Bitboard magic_helpers::blockmask_bishop(Square square)
    {
        Bitboard edges = Bitboards::a_file | Bitboards::h_file | Bitboards::rank_1 | Bitboards::rank_8;
        Bitboard cross = pseudo_attacks_bishops(square);
        return cross & (~edges);
    }

    Bitboard magic_helpers::moveboard_rook(Square square, Bitboard blockboard)
    {
        int di[4] = { -1, 1, 0, 0 };
        int dj[4] = { 0, 0, -1, 1 };
        int file = square % 8;
        int rank = square / 8;
        Bitboard result;
        for (int dir = 0; dir < 4; dir++)
        {
            for (int k = 1; k < 8; k++)
            {
                int i = rank + k * di[dir];
                int j = file + k * dj[dir];
                if (!inside_board(i, j))
                    break;
                int index = i * 8 + j;
                result.set(index);
                if (blockboard.test(index))
                    break;
            }
        }
        return result;
    }

    Bitboard magic_helpers::moveboard_bishop(Square square, Bitboard blockboard)
    {
        int di[4] = { -1, -1, 1, 1 };
        int dj[4] = { -1, 1, -1, 1 };
        int file = square % 8;
        int rank = square / 8;
        Bitboard result;
        for (int dir = 0; dir < 4; dir++)
        {
            for (int k = 1; k < 8; k++)
            {
                int i = rank + k * di[dir];
                int j = file + k * dj[dir];
                if (!inside_board(i, j))
                    break;
                int index = i * 8 + j;
                result.set(index);
                if (blockboard.test(index))
                    break;
            }
        }
        return result;
    }

    std::vector<Bitboard> magic_helpers::gen_blockboards(Bitboard blockmask)
    {
        int n_bits = blockmask.count();
        int n_blockers = 1 << n_bits;

        std::vector<Bitboard> single_bits(n_bits);
        for (int i = 0; i < n_bits; i++)
            single_bits[i] = Bitboard::from_single_bit(blockmask.bitscan_forward_reset());

        std::vector<Bitboard> result(n_blockers);
        for (int i = 0; i < n_blockers; i++)
            for (int j = 0; j < n_bits; j++)
                if ((i & (1 << j)) > 0)
                    result[i] += single_bits[j];

        return result;
    }

    uint64_t magic_helpers::gen_magic(Bitboard blockmask, std::vector<Bitboard> blockboards)
    {
        int n_bits = blockmask.count();
        std::vector<uint64_t> numbers(blockboards.size());

        uint64_t result;
        bool unique = false;

        while (!unique)
        {
            result = random_uint64_fewbits();
            for (unsigned int i = 0; i < blockboards.size(); i++)
            {
                unique = true;
                numbers[i] = (blockboards[i].to_uint64() * result) >> (64 - n_bits);
                for (unsigned int j = 0; j < i; j++)
                {
                    unique = (numbers[i] != numbers[j]);
                    if (!unique)
                        break;
                }
                if (!unique)
                    break;
            }
        }

        return result;
    }

    uint64_t magic_helpers::random_uint64()
    {
        uint64_t u1 = rand() & 0xFFFF;
        uint64_t u2 = rand() & 0xFFFF;
        uint64_t u3 = rand() & 0xFFFF;
        uint64_t u4 = rand() & 0xFFFF;
        return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
    }
    uint64_t magic_helpers::random_uint64_fewbits()
    {
        return random_uint64() & random_uint64() & random_uint64();
    }


    void magic_helpers::gen_all_magics(bool compute)
    {
        // Bishops
        for (int square = 0; square < 64; square++)
        {
            Bitboard blockmask = blockmask_bishop(static_cast<Square>(square));
            std::vector<Bitboard> blockboards = gen_blockboards(blockmask);
            std::vector<Bitboard> moveboards(blockboards.size());
            for (unsigned int i = 0; i < blockboards.size(); i++)
                moveboards[i] = moveboard_bishop(static_cast<Square>(square), blockboards[i]);
            uint64_t magic;
            if (compute)
                magic = gen_magic(blockmask, blockboards);
            else
                magic = magic_helpers::bishop_magics[square];
            Bitboards::bishop_magics[square] = MagicBitboard(magic, blockmask, blockboards, moveboards);
        }
        // Rooks
        for (int square = 0; square < 64; square++)
        {
            Bitboard blockmask = blockmask_rook(static_cast<Square>(square));
            std::vector<Bitboard> blockboards = gen_blockboards(blockmask);
            std::vector<Bitboard> moveboards(blockboards.size());
            for (unsigned int i = 0; i < blockboards.size(); i++)
                moveboards[i] = moveboard_rook(static_cast<Square>(square), blockboards[i]);
            uint64_t magic;
            if (compute)
                magic = gen_magic(blockmask, blockboards);
            else
                magic = magic_helpers::rook_magics[square];
            Bitboards::rook_magics[square] = MagicBitboard(magic, blockmask, blockboards, moveboards);
        }
    }
}
