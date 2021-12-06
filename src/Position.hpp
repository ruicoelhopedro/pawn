#pragma once

#include "Types.hpp"
#include "Bitboard.hpp"
#include "Move.hpp"
#include "Zobrist.hpp"
#include "PieceSquareTables.hpp"
#include <algorithm>
#include <string>

enum class MoveGenType
{
    LEGAL,
    QUIETS,
    CAPTURES
};

class Board
{
    // Required fields
    Bitboard m_pieces[NUM_PIECE_TYPES][NUM_COLORS];
    Turn m_turn;
    bool m_castling_rights[NUM_CASTLE_SIDES][NUM_COLORS];
    Square m_enpassant_square;
    int m_half_move_clock;
    int m_full_move_clock;

    // Updated fields
    Hash m_hash;
    Bitboard m_checkers;
    MixedScore m_eval1;
    int m_phase;
    BoardPieces m_board_pieces[NUM_SQUARES];

protected:

    template<Turn TURN, Piece PIECE_TYPE>
    void generate_moves(MoveList& list, Bitboard filter, Bitboard occupancy) const
    {
        static_assert(PIECE_TYPE != PAWN && PIECE_TYPE != KING, "Pawn and king not supported!");

        Bitboard pieces = m_pieces[PIECE_TYPE][TURN];

        while (pieces)
        {
            Square from = pieces.bitscan_forward_reset();
            Bitboard attacks = Bitboards::get_attacks<PIECE_TYPE>(from, occupancy) & filter;

            while (attacks)
            {
                Square to = attacks.bitscan_forward_reset();
                list.push(from, to, occupancy.test(to) ? CAPTURE : QUIET);
            }
        }
    }


    template<Turn TURN>
    void generate_moves_pawns(MoveList& list, Bitboard filter, Bitboard occupancy) const
    {
        constexpr Bitboard rank3 = (TURN == WHITE) ? Bitboards::rank_3 : Bitboards::rank_6;
        constexpr Bitboard rank7 = (TURN == WHITE) ? Bitboards::rank_7 : Bitboards::rank_2;

        constexpr Direction up = (TURN == WHITE) ? 8 : -8;
        constexpr Direction left = -1;
        constexpr Direction right = -left;

        Bitboard enemy_pieces = get_pieces<~TURN>() & filter;
        Bitboard empty_squares = ~occupancy;

        Bitboard pawns = get_pieces<TURN, PAWN>();
        Bitboard promoting_pawns = pawns & rank7;
        Bitboard non_promoting_pawns = pawns & ~rank7;

        // Single and double pawn pushes
        Bitboard single_pushes = non_promoting_pawns.shift<up>() & empty_squares;
        Bitboard double_pushes = (single_pushes & rank3).shift<up>() & empty_squares & filter;
        single_pushes &= filter;

        while (single_pushes)
        {
            Square to = single_pushes.bitscan_forward_reset();
            list.push(to - up, to);
        }
        while (double_pushes)
        {
            Square to = double_pushes.bitscan_forward_reset();
            list.push(to - up - up, to, DOUBLE_PAWN_PUSH);
        }

        // Captures (non ep)
        Bitboard left_captures  = (non_promoting_pawns & ~Bitboards::a_file).shift<up + left >() & enemy_pieces;
        Bitboard right_captures = (non_promoting_pawns & ~Bitboards::h_file).shift<up + right>() & enemy_pieces;
        while (left_captures)
        {
            Square to = left_captures.bitscan_forward_reset();
            list.push(to - (up - 1), to, CAPTURE);
        }
        while (right_captures)
        {
            Square to = right_captures.bitscan_forward_reset();
            list.push(to - (up + 1), to, CAPTURE);
        }

        // En passant: test if the captured pawn is in the filter
        if (m_enpassant_square != SQUARE_NULL && filter.test(m_enpassant_square - up))
        {
            Square king_square = get_pieces<TURN, KING>().bitscan_forward();
            Bitboard ep_attackers = Bitboards::get_attacks_pawns<~TURN>(m_enpassant_square) & non_promoting_pawns;
            while (ep_attackers)
            {
                // Check if king is in the same rank of the ep capture
                Square from = ep_attackers.bitscan_forward_reset();
                if (Bitboards::ranks[rank(from)].test(king_square))
                {
                    // Is there an enemy rook or queen in the same rank?
                    Bitboard rooks_queens = (get_pieces<~TURN, ROOK>() | get_pieces<~TURN, QUEEN>()) & Bitboards::ranks[rank(from)];
                    if (rooks_queens)
                    {
                        // Remove attacker and captured pawn from occupancy and test for check
                        Bitboard new_occupancy = occupancy;
                        new_occupancy.reset(from);
                        new_occupancy.reset(m_enpassant_square - up);
                        bool check = false;
                        while (rooks_queens && !check)
                            check = Bitboards::get_attacks<ROOK>(rooks_queens.bitscan_forward_reset(), new_occupancy).test(king_square);

                        // Skip this ep-capture if it leaves our king in check
                        if (check)
                            continue;
                    }
                }
                list.push(from, m_enpassant_square, EP_CAPTURE);
            }
        }

        // Promotions
        Bitboard forward_promo = promoting_pawns.shift<up>() & empty_squares & filter;
        Bitboard left_capture_promo  = (promoting_pawns & ~Bitboards::a_file).shift<up + left >() & enemy_pieces & filter;
        Bitboard right_capture_promo = (promoting_pawns & ~Bitboards::h_file).shift<up + right>() & enemy_pieces & filter;
        while (forward_promo)
        {
            Square to = forward_promo.bitscan_forward_reset();
            list.push_promotions<false>(to - up, to);
        }
        while (left_capture_promo)
        {
            Square to = left_capture_promo.bitscan_forward_reset();
            list.push_promotions<true>(to - (up + left), to);
        }
        while (right_capture_promo)
        {
            Square to = right_capture_promo.bitscan_forward_reset();
            list.push_promotions<true>(to - (up + right), to);
        }
    }


    template<Turn TURN>
    void generate_moves_king(MoveList& list, Bitboard filter, Bitboard occupancy) const
    {
        Square king_square = get_pieces<TURN, KING>().bitscan_forward();
        Bitboard attacks = Bitboards::get_attacks<KING>(king_square, occupancy) & filter;
        Bitboard occupancy_noking = occupancy ^ get_pieces<TURN, KING>();

        while (attacks)
        {
            // Test if target square is attacked
            Square target = attacks.bitscan_forward_reset();
            if (!attackers<~TURN>(target, occupancy_noking))
                list.push(king_square, target, occupancy.test(target) ? CAPTURE : QUIET);
        }

        // Castling when not in check
        if (!checkers())
        {
            if (m_castling_rights[KINGSIDE][TURN] && filter.test(Bitboards::castle_target_square[TURN][KINGSIDE]) && can_castle<TURN>(KINGSIDE, occupancy))
                list.push(king_square, Bitboards::castle_target_square[TURN][KINGSIDE], KING_CASTLE);
            if (m_castling_rights[QUEENSIDE][TURN] && filter.test(Bitboards::castle_target_square[TURN][QUEENSIDE]) && can_castle<TURN>(QUEENSIDE, occupancy))
                list.push(king_square, Bitboards::castle_target_square[TURN][QUEENSIDE], QUEEN_CASTLE);
        }

    }


    template<Turn TURN>
    void generate_moves(MoveList& list, MoveGenType type) const
    {
        Bitboard occupancy = get_pieces();
        Square king_square = get_pieces<TURN, KING>().bitscan_forward();
        Bitboard pinners;
        Bitboard pinned = pins<TURN>(king_square, occupancy, pinners);

        Bitboard filter;
        if (type == MoveGenType::LEGAL)
            filter = ~get_pieces<TURN>();
        else if (type == MoveGenType::QUIETS)
            filter = ~occupancy;
        else if (type == MoveGenType::CAPTURES)
            filter = get_pieces<~TURN>();

        // Not a double check
        if (!checkers().more_than_one())
        {
            Bitboard new_filter = filter;
            if (checkers())
                new_filter &= (checkers() | Bitboards::between(king_square, checkers().bitscan_forward()));

            // Moves for each piece type (except king)
            generate_moves_pawns<TURN  >(list, new_filter, occupancy);
            generate_moves<TURN, KNIGHT>(list, new_filter, occupancy);
            generate_moves<TURN, BISHOP>(list, new_filter, occupancy);
            generate_moves<TURN, ROOK  >(list, new_filter, occupancy);
            generate_moves<TURN, QUEEN >(list, new_filter, occupancy);
        }

        // King moves: always legal
        generate_moves_king<TURN>(list, filter, occupancy);

        // Check for pins
        if (pinned)
        {
            auto move = list.begin();
            // Iterate list and pop the move if:
            // 1. Piece is pinned
            // 2. The piece can't move to the destination without breaking the pin (or capturing the pinner)
            while (move != list.end())
                if (pinned.test(move->from()) && !test_pinned_move<TURN>(move->from(), move->to(), pinners, king_square))
                    list.pop(move);
                else
                    move++;
        }
    }


    template<Turn TURN>
    bool test_pinned_move(Square pinned, Square target, Bitboard pinners, Square king_square) const
    {
        while (pinners)
        {
            // Check if both the source and destination squares are between the king and pinner or capture the pinner
            Square pinner = pinners.bitscan_forward_reset();
            Bitboard legals = Bitboards::between(pinner, king_square) | Bitboard::from_single_bit(pinner);
            if (legals.test(pinned) && legals.test(target))
                return true;
        }
        return false;
    }


    void init();


    Hash generate_hash() const;


    void update_checkers();


    template<bool IS_NEW>
    void set_piece(Piece piece, Turn turn, Square square)
    {
        PieceType pt = static_cast<PieceType>(piece);
        m_pieces[piece][turn].set(square);
        m_hash ^= Zobrist::get_piece_turn_square(pt, turn, square);
        m_board_pieces[square] = get_board_piece(piece, turn);
        m_eval1 += piece_square(piece, square, turn) * turn_to_color(turn);
        if (IS_NEW)
        {
            m_eval1 += piece_value[piece] * turn_to_color(turn);
            m_phase -= Phases::Pieces[piece];
        }
    }


    template<bool IS_NEW>
    void pop_piece(Piece piece, Turn turn, Square square)
    {
        PieceType pt = static_cast<PieceType>(piece);
        m_pieces[piece][turn].reset(square);
        m_hash ^= Zobrist::get_piece_turn_square(pt, turn, square);
        m_board_pieces[square] = NO_PIECE;
        m_eval1 -= piece_square(piece, square, turn) * turn_to_color(turn);
        if (IS_NEW)
        {
            m_eval1 -= piece_value[piece] * turn_to_color(turn);
            m_phase += Phases::Pieces[piece];
        }
    }


    void unset_castling(CastleSide side, Turn turn);


    template<Turn TURN>
    void update_checkers()
    {
        Bitboard king_bb = get_pieces<TURN, KING>();
        Square king_square = king_bb.bitscan_forward();
        m_checkers = attackers<~TURN>(king_square, get_pieces());
    }


    template<Turn TURN, Piece PIECE_TYPE>
    bool legal(Move move, Bitboard occupancy) const
    {
        Bitboard filter = ~get_pieces<TURN>();
        Square king_square = get_pieces<TURN, KING>().bitscan_forward();
        if (checkers())
            filter = (checkers() | Bitboards::between(king_square, checkers().bitscan_forward()));

        // Double check and non-king move
        if (checkers().more_than_one() && PIECE_TYPE != KING)
            return false;

        // Per piece logic
        if (PIECE_TYPE == PAWN)
        {
            constexpr Bitboard rank2 = (TURN == WHITE) ? Bitboards::rank_2 : Bitboards::rank_7;
            constexpr Bitboard rank7 = (TURN == WHITE) ? Bitboards::rank_7 : Bitboards::rank_2;
            constexpr Direction up_vec = (TURN == WHITE) ? 1 : -1;
            constexpr Direction up = (TURN == WHITE) ? 8 : -8;
            constexpr Direction left = -1;
            constexpr Direction right = -left;

            // Quick check on move direction and square distance
            int dist = up_vec * (move.to() - move.from());
            if ((dist < 7) || (dist > 9 && dist != 16))
                return false;

            // Promotions on non-rank 7
            if (rank7.test(move.from()) != move.is_promotion())
                return false;

            // Short double pawn pushes
            if (dist < 16 && move.is_double_pawn_push())
                return false;

            if (dist == 8)
            {
                // Single pushes
                if (move.is_capture())
                    return false;
                if (occupancy.test(move.to()) || !filter.test(move.to()))
                    return false;
            }
            else if (dist == 16)
            {
                // Double pushes
                if (!rank2.test(move.from()) || occupancy.test(move.to()) || occupancy.test(move.to() - up) || !filter.test(move.to()))
                    return false;
                if (!move.is_double_pawn_push())
                    return false;
            }
            else
            {
                // Captures
                if (!move.is_capture())
                    return false;

                // Enemy pieces (with ep)
                Bitboard enemy_pieces = get_pieces<~TURN>() & filter;
                if (m_enpassant_square != SQUARE_NULL && filter.test(m_enpassant_square - up))
                    enemy_pieces.set(m_enpassant_square);

                // Generate and test captures for this pawn
                Bitboard pawn = Bitboard::from_single_bit(move.from());
                Bitboard left_captures = (pawn & ~Bitboards::a_file).shift<up + left >() & enemy_pieces;
                Bitboard right_captures = (pawn & ~Bitboards::h_file).shift<up + right>() & enemy_pieces;
                if (!left_captures.test(move.to()) && !right_captures.test(move.to()))
                    return false;

                // Is ep capture?
                if (move.to() == m_enpassant_square)
                {
                    if (!move.is_ep_capture())
                        return false;

                    // King in the same rank?
                    if (Bitboards::ranks[rank(move.from())].test(king_square))
                    {
                        // Is there an enemy rook or queen in the same rank?
                        Bitboard rooks_queens = (get_pieces<~TURN, ROOK>() | get_pieces<~TURN, QUEEN>()) & Bitboards::ranks[rank(move.from())];
                        if (rooks_queens)
                        {
                            // Remove attacker and captured pawn from occupancy and test for check
                            Bitboard new_occupancy = occupancy;
                            new_occupancy.reset(move.from());
                            new_occupancy.reset(m_enpassant_square - up);
                            while (rooks_queens)
                                if (Bitboards::get_attacks<ROOK>(rooks_queens.bitscan_forward_reset(), new_occupancy).test(king_square))
                                    return false;
                        }
                    }
                }
            }
        }
        else if (PIECE_TYPE == KING)
        {
            Bitboard occupancy_noking = occupancy ^ get_pieces<TURN, KING>();
            Bitboard attacks = Bitboards::get_attacks<PIECE_TYPE>(move.from(), occupancy);
            // If move pseudolegal, return whether the target square is attacked or not
            if (!move.is_castle() && attacks.test(move.to()))
                return !attackers<~TURN>(move.to(), occupancy_noking);

            // Castling test
            if (!checkers() && move.is_castle())
            {
                // Starting square
                if (move.from() != (TURN == WHITE ? SQUARE_E1 : SQUARE_E8))
                    return false;
                // Kingside
                if (m_castling_rights[KINGSIDE][TURN] && move.move_type() == KING_CASTLE &&
                    move.to() == Bitboards::castle_target_square[TURN][KINGSIDE] &&
                    can_castle<TURN>(KINGSIDE, occupancy))
                    return true;
                // Queenside
                if (m_castling_rights[QUEENSIDE][TURN] && move.move_type() == QUEEN_CASTLE &&
                    move.to() == Bitboards::castle_target_square[TURN][QUEENSIDE] &&
                    can_castle<TURN>(QUEENSIDE, occupancy))
                    return true;
            }

            return false;
        }
        else
        {
            Bitboard attacks = Bitboards::get_attacks<PIECE_TYPE>(move.from(), occupancy) & filter;
            if (!attacks.test(move.to()))
                return false;
        }

        // Pinned move test
        Bitboard pinners;
        Bitboard pinned = pins<TURN>(king_square, occupancy, pinners);
        return !(pinned.test(move.from()) && !test_pinned_move<TURN>(move.from(), move.to(), pinners, king_square));
    }


    template<Piece PIECE_TYPE>
    bool legal(Move move, Bitboard occupancy) const
    {
        if (m_turn == WHITE)
            return legal<WHITE, PIECE_TYPE>(move, occupancy);
        else
            return legal<BLACK, PIECE_TYPE>(move, occupancy);
    }

public:
    Board();


    Board(std::string fen);


    friend std::ostream& operator<<(std::ostream& out, const Board& board);


    std::string to_fen() const;


    Board make_move(Move move) const;


    Board make_null_move();


    int half_move_clock() const;


    void generate_moves(MoveList& list, MoveGenType type) const;


    inline PieceType get_piece_at(Square square) const
    {
        return static_cast<PieceType>(get_piece(m_board_pieces[square]));
    }


    bool is_valid() const;


    template<Turn TURN, Piece PIECE_TYPE>
    Bitboard get_pieces() const { return m_pieces[PIECE_TYPE][TURN]; }


    template<Turn TURN>
    Bitboard get_pieces() const
    {
        return m_pieces[PAWN  ][TURN]
             | m_pieces[KNIGHT][TURN]
             | m_pieces[BISHOP][TURN]
             | m_pieces[ROOK  ][TURN]
             | m_pieces[QUEEN ][TURN]
             | m_pieces[KING  ][TURN];
    }


    Bitboard get_pieces() const;


    Bitboard get_pieces(Turn turn, Piece piece) const;


    Bitboard checkers() const;


    Turn turn() const;


    template<Turn TURN>
    Bitboard attackers(Square square, Bitboard occupancy) const
    {
        return (Bitboards::get_attacks_pawns<~TURN>(square)       &  get_pieces<TURN, PAWN  >())                              |
               (Bitboards::get_attacks<KNIGHT>(square, occupancy) &  get_pieces<TURN, KNIGHT>())                              |
               (Bitboards::get_attacks<BISHOP>(square, occupancy) & (get_pieces<TURN, BISHOP>() | get_pieces<TURN, QUEEN>())) |
               (Bitboards::get_attacks<ROOK  >(square, occupancy) & (get_pieces<TURN, ROOK  >() | get_pieces<TURN, QUEEN>())) |
               (Bitboards::get_attacks<KING  >(square, occupancy) &  get_pieces<TURN, KING  >());
    }


    template<Turn TURN>
    Bitboard attackers_battery(Square square, Bitboard occupancy) const
    {
        Bitboard bishops = get_pieces<TURN, BISHOP>() | get_pieces<TURN, QUEEN>();
        Bitboard rooks = get_pieces<TURN, ROOK>() | get_pieces<TURN, QUEEN>();
        return (Bitboards::get_attacks_pawns<~TURN>(square)                 & get_pieces<TURN, PAWN  >()) |
               (Bitboards::get_attacks<KNIGHT>(square, occupancy)           & get_pieces<TURN, KNIGHT>()) |
               (Bitboards::get_attacks<BISHOP>(square, occupancy ^ bishops) & bishops)                    |
               (Bitboards::get_attacks<ROOK  >(square, occupancy ^ rooks)   & rooks)                      |
               (Bitboards::get_attacks<KING  >(square, occupancy)           & get_pieces<TURN, KING>());
    }


    Bitboard attackers(Square square, Bitboard occupancy, Turn turn) const;


    template<Turn TURN>
    Bitboard pins(Square square, Bitboard occupancy, Bitboard& pinners) const
    {
        // Select sliding pieces from opponent
        Bitboard bishops = m_pieces[BISHOP][~TURN] | m_pieces[QUEEN][~TURN];
        Bitboard rooks   = m_pieces[ROOK][~TURN]   | m_pieces[QUEEN][~TURN];

        // Select candidates for pinning one of our pieces
        Bitboard pinner_candidates = (Bitboards::diagonals[square]   & bishops) |
                                     (Bitboards::ranks_files[square] & rooks);

        // Build an occupancy excluding pinner candidates
        Bitboard occupancy_excluded = occupancy ^ pinner_candidates;

        Bitboard pinned;
        pinners = Bitboard();

        // Iterate over possible pinners
        while (pinner_candidates)
        {
            // Build a bitboard with pieces between target square and pinner
            int pinner = pinner_candidates.bitscan_forward_reset();
            Bitboard pieces_between = Bitboards::between(square, pinner) & occupancy_excluded;

            // Check if it is a pin
            if (pieces_between && !pieces_between.more_than_one())
            {
                // Yep, flag it and store pinner
                pinned |= pieces_between;
                pinners.set(pinner);
            }
        }

        return pinned;
    }


    template<Turn TURN>
    bool can_castle(CastleSide side, Bitboard occupancy) const
    {
        // Check if target squares are empty
        if (occupancy & Bitboards::castle_non_occupied_squares[TURN][side])
            return false;

        // Check if middle squares are attacked
        Bitboard attackable = Bitboards::castle_non_attacked_squares[TURN][side];
        while (attackable)
            if (attackers<~TURN>(attackable.bitscan_forward_reset(), occupancy))
                return false;

        return true;
    }


    bool operator==(const Board& other) const;


    Hash hash() const;


    Square least_valuable(Bitboard bb) const;


    Score see(Move move, int threshold = 0) const;


    MixedScore material_eval() const;


    uint8_t phase() const;


    bool legal(Move move) const;


    Bitboard non_pawn_material() const;


    Bitboard non_pawn_material(Turn turn) const;


    Bitboard sliders() const;
};



class Position
{
    std::vector<Board> m_boards;
    MoveStack m_stack;

public:
    Position();


    Position(std::string fen);


    bool is_draw(bool unique) const;


    bool is_check() const;


    Turn get_turn() const;


    MoveList generate_moves(MoveGenType type);


    void make_move(Move move);


    void unmake_move();


    void make_null_move();


    void unmake_null_move();


    Board& board();


    const Board& board() const;


    Hash hash() const;


    MoveList move_list() const;


    void set_init_ply();
};


std::ostream& operator<<(std::ostream& out, const Board& board);
