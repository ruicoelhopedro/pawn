#include "Position.hpp"
#include "Types.hpp"
#include "NNUE.hpp"
#include "UCI.hpp"
#include "Zobrist.hpp"
#include "syzygy/syzygy.hpp"
#include <cassert>
#include <sstream>
#include <string>
#include <cctype>
#include <cstring>


PieceType fen_piece(char c)
{
    char lower = tolower(c);
    return lower == 'p' ? PAWN
         : lower == 'n' ? KNIGHT
         : lower == 'b' ? BISHOP
         : lower == 'r' ? ROOK
         : lower == 'q' ? QUEEN
         : lower == 'k' ? KING
         : PIECE_NONE;
}


char fen_piece(Piece pc)
{
    PieceType piece = get_piece_type(pc);
    char p = piece == PAWN   ? 'p'
           : piece == KNIGHT ? 'n'
           : piece == BISHOP ? 'b'
           : piece == ROOK   ? 'r'
           : piece == QUEEN  ? 'q'
           : piece == KING   ? 'k'
           : 'x';
    return get_turn(pc) == WHITE ? toupper(p) : p;
}


CastleSide fen_castle_side(char c)
{
    char lower = tolower(c);
    return lower == 'k' ? KINGSIDE
         : lower == 'q' ? QUEENSIDE
         : NO_SIDE;
}


CastleFile fen_castle_file_chess960(char c)
{
    char lower = tolower(c);
    return lower == 'a' ? CastleFile::FILE_A
         : lower == 'b' ? CastleFile::FILE_B
         : lower == 'c' ? CastleFile::FILE_C
         : lower == 'd' ? CastleFile::FILE_D
         : lower == 'e' ? CastleFile::FILE_E
         : lower == 'f' ? CastleFile::FILE_F
         : lower == 'g' ? CastleFile::FILE_G
         : lower == 'h' ? CastleFile::FILE_H
         : CastleFile::NONE;
}


char fen_castle_side(CastleSide side, Turn turn)
{
    char c = side == KINGSIDE  ? 'k'
           : side == QUEENSIDE ? 'q'
           : 'x';
    return turn == WHITE ? toupper(c) : c;
}


char fen_castle_side_chess960(CastleFile side, Turn turn)
{
    char c = "xabcdefgh"[static_cast<int>(side)];
    return turn == WHITE ? toupper(c) : c;
}


Board::Board()
    : Board(UCI::Options::UCI_Chess960 ? "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w HAha - 0 1"
                                       : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
{}


Board::Board(std::string fen)
    : m_hash(0)
{
    auto c = fen.cbegin();

    // Default initialisation for board pieces
    std::memset(m_board_pieces, PIECE_NONE, sizeof(m_board_pieces));
    std::memset(m_king_sq, 0, sizeof(m_king_sq));

    // Read position
    Square square = SQUARE_A8;
    while (c < fen.cend() && !isspace(*c))
    {
        if (isdigit(*c))
            square += *c - '0';
        else if (*c == '/')
            square -= 16;
        else
            set_piece(fen_piece(*c), isupper(*c) ? WHITE : BLACK, square++);

        c++;
    }
    m_king_sq[WHITE] = m_pieces[KING][WHITE].bitscan_forward();
    m_king_sq[BLACK] = m_pieces[KING][BLACK].bitscan_forward();

    // Side to move
    m_turn = WHITE;
    while ((++c) < fen.cend() && !isspace(*c))
        m_turn = (*c == 'w') ? WHITE : BLACK;

    // Castling rights
    std::memset(m_castling_rights, 0, sizeof(m_castling_rights));
    while ((++c) < fen.cend() && !isspace(*c))
    {
        if (UCI::Options::UCI_Chess960)
        {
            char lower = tolower(*c);
            if (lower == 'k' || lower == 'q')
            {
                // X-FEN notation: find available rook
                Turn turn = isupper(*c) ? WHITE : BLACK;
                Direction dir = lower == 'k' ? 1 : -1;
                for (int f = file(m_king_sq[turn]); f >= 0 && f < 8; f += dir)
                    if (get_piece_type(m_board_pieces[make_square(turn == WHITE ? 0 : 7, f)]) == ROOK)
                        set_castling(lower == 'k' ? KINGSIDE : QUEENSIDE, turn, fen_castle_file_chess960('a' + f));
            }
            else if (lower >= 'a' && lower <= 'h')
            {
                // Shredder-FEN notation
                Turn turn = isupper(*c) ? WHITE : BLACK;
                CastleFile file = fen_castle_file_chess960(*c);
                set_castling(get_rook_square(file, turn) > m_king_sq[turn] ? KINGSIDE : QUEENSIDE, turn, file);
            }
        }
        else
        {
            CastleSide side = fen_castle_side(*c);
            if (side != NO_SIDE)
                set_castling(side, isupper(*c) ? WHITE : BLACK, side == KINGSIDE ? CastleFile::FILE_H : CastleFile::FILE_A);
        }
    }
              

    // Ep square
    m_enpassant_square = SQUARE_NULL;
    while ((++c) < fen.cend() && !isspace(*c))
        if (*c != '-' && (++c) != fen.cend() && !isspace(*c))
            m_enpassant_square = make_square(*c - '1', *(c-1) - 'a');

    // Half-move clock
    m_half_move_clock = 0;
    while ((++c) < fen.cend() && !isspace(*c))
        m_half_move_clock = m_half_move_clock * 10 + (*c - '0');

    // Full-move clock
    m_full_move_clock = 0;
    while ((++c) < fen.cend() && !isspace(*c))
        m_full_move_clock = m_full_move_clock * 10 + (*c - '0');
    m_full_move_clock = std::max(m_full_move_clock, 1);

    // Update remaining hash: turn and ep square
    if (m_turn == Turn::BLACK)
        m_hash ^= Zobrist::get_black_move();
    if (m_enpassant_square != SQUARE_NULL)
        m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    update_checkers();

    regen_psqt(WHITE);
    regen_psqt(BLACK);
}


std::string Board::to_fen() const
{
    std::ostringstream ss;

    // Position
    for (int rank = 7; rank >= 0; rank--)
    {
        int space = 0;
        for (int file = 0; file < 8; file++)
        {
            Piece pc = m_board_pieces[make_square(rank, file)];
            if (pc == Piece::NO_PIECE)
                space++;
            else
            {
                if (space)
                    ss << space;
                ss << fen_piece(pc);
                space = 0;
            }
        }
        if (space)
            ss << space;
        ss << (rank > 0 ? '/' : ' ');
    }

    // Side to move
    ss << (m_turn == WHITE ? "w " : "b ");

    // Castling rights
    bool found = false;
    for (auto turn : { WHITE, BLACK })
        for (auto side : { KINGSIDE, QUEENSIDE })
            if (castling_rights(side, turn))
            {
                found = true;
                ss << (UCI::Options::UCI_Chess960 ? fen_castle_side_chess960(m_castling_rights[side][turn], turn)
                                                  : fen_castle_side(side, turn));
            }
    ss << (found ? " " : "- ");

    // Ep square
    ss << (m_enpassant_square == SQUARE_NULL ? "-" : get_square(m_enpassant_square)) << ' ';

    // Half- and full-move clocks
    ss << m_half_move_clock << ' ' << m_full_move_clock;

    return ss.str();
}


Hash Board::generate_hash() const
{
    Hash hash = 0;

    // Position hash
    for (Turn turn : { WHITE, BLACK })
        for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        {
            Bitboard piece_bb = m_pieces[piece][turn];
            while (piece_bb)
                hash ^= Zobrist::get_piece_turn_square(piece, turn, piece_bb.bitscan_forward_reset());
        }

    // Turn to move
    if (m_turn == BLACK)
        hash ^= Zobrist::get_black_move();

    // En-passsant square
    if (m_enpassant_square != SQUARE_NULL)
        hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    // Castling rights
    for (CastleSide side : { KINGSIDE, QUEENSIDE })
        for (Turn turn : { WHITE, BLACK })
            if (castling_rights(side, turn))
                hash ^= Zobrist::get_castle_side_turn(side, turn);

    return hash;
}


void Board::update_checkers()
{
    if (m_turn == WHITE)
        update_checkers<WHITE>();
    else
        update_checkers<BLACK>();
}


void Board::regen_psqt(Turn turn)
{
    m_acc[turn].clear();
    std::size_t num_features = 0;
    NNUE::Feature features[NNUE::NUM_MAX_ACTIVE_FEATURES];

    // Pack features to be pushed into the accumulator
    for (Turn t : { WHITE, BLACK })
        for (PieceType p : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
        {
            Bitboard b = get_pieces(t, p);
            while (b)
                features[num_features++] = m_acc[turn].get_feature(p, b.bitscan_forward_reset(), m_king_sq[turn], t, turn);
        }

    // Regen entire accumulator at once
    m_acc[turn].push_features(num_features, features);
}


void Board::generate_moves(MoveList& list, MoveGenType type) const
{
    if (m_turn == WHITE)
        generate_moves<WHITE>(list, type);
    else
        generate_moves<BLACK>(list, type);
}


Board Board::make_move(Move move) const
{
    Board result = *this;
    const Direction up = (m_turn == WHITE) ? 8 : -8;
    const PieceType piece = get_piece_at(move.from());

    // Increment clocks
    result.m_full_move_clock += m_turn;
    if (piece == PAWN || move.is_capture())
        result.m_half_move_clock = 0;
    else
        result.m_half_move_clock++;

    // Initial empty ep square
    result.m_enpassant_square = SQUARE_NULL;

    // Per move type action
    if (move.is_capture())
    {
        // Captured square is different for ep captures
        Square target = move.is_ep_capture() ? move.to() - up : move.to();

        // Remove captured piece
        result.pop_piece(get_piece_at(target), ~m_turn, target);

        // Castling: check if any rook has been captured
        for (auto side : { KINGSIDE, QUEENSIDE })
            if (castling_rights(side, ~m_turn) && move.to() == get_rook_square(m_castling_rights[side][~m_turn], ~m_turn))
                result.unset_castling(side, ~m_turn);
    }
    else if (move.is_double_pawn_push())
    {
        // Update ep square
        result.m_enpassant_square = move.to() - up;
        result.m_hash ^= Zobrist::get_ep_file(file(move.to()));
    }
    else if (move.is_castle())
    {
        // Get the start and ending squares for the rook
        CastleSide side = file(move.to()) >= 4 ? KINGSIDE : QUEENSIDE;
        Square iS = get_rook_square(m_castling_rights[side][m_turn], m_turn);
        Square iE = move.to() + (side == KINGSIDE ? -1 : +1);

        // Make the move in stages to ensure correct updates in Chess960
        if (UCI::Options::UCI_Chess960)
        {
            result.pop_piece(ROOK, m_turn, iS);
            result.move_piece(piece, m_turn, move.from(), move.to());
            result.set_piece(ROOK, m_turn, iE);
        }
        else
        {
            result.move_piece(piece, m_turn, move.from(), move.to());
            result.move_piece(ROOK, m_turn, iS, iE);
        }
    }

    // Set piece on target square
    if (move.is_promotion())
    {
        result.pop_piece(piece, m_turn, move.from());
        result.set_piece(move.promo_piece(), m_turn, move.to());
    }
    else if (!move.is_castle())
    {
        result.move_piece(piece, m_turn, move.from(), move.to());
    }

    // After a king move, update PSQ tables
    if (piece == KING)
    {
        result.m_king_sq[m_turn] = result.m_pieces[KING][m_turn].bitscan_forward();
        result.regen_psqt(m_turn);
    }

    // Update castling rights after this move
    if (piece == KING)
    {
        // Unset all castling rights after a king move
        for (auto side : { KINGSIDE, QUEENSIDE })
            result.unset_castling(side, m_turn);
    }
    else if (piece == ROOK)
    {
        // Unset castling rights for a certain side if a rook moves
        if (move.from() == get_rook_square(m_castling_rights[KINGSIDE][m_turn], m_turn))
            result.unset_castling(KINGSIDE, m_turn);
        if (move.from() == get_rook_square(m_castling_rights[QUEENSIDE][m_turn], m_turn))
            result.unset_castling(QUEENSIDE, m_turn);
    }

    // Swap turns
    result.m_turn = ~m_turn;
    result.m_hash ^= Zobrist::get_black_move();

    // Reset previous en-passant hash
    if (m_enpassant_square != SQUARE_NULL)
        result.m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    // Update checkers
    result.update_checkers();

    return result;
}


Hash Board::hash_after(Move move) const
{
    Hash result = hash() ^ Zobrist::get_black_move();

    // Moved piece (and handle promotions)
    PieceType piece = get_piece_at(move.from());
    PieceType target_piece = move.is_promotion() ? move.promo_piece() : piece;
    result ^= Zobrist::get_piece_turn_square(piece, turn(), move.from());
    result ^= Zobrist::get_piece_turn_square(target_piece, turn(), move.to());

    // Handle special moves
    if (move.is_capture())
    {
        // Remove captured piece, if any
        Square target = move.is_ep_capture() ? move.to() - (turn() == WHITE ? 8 : -8) : move.to();
        PieceType captured = get_piece_at(target);
        result ^= Zobrist::get_piece_turn_square(captured, ~turn(), target);

        // Castling: check if any rook has been captured
        if (captured == ROOK)
            for (auto side : { KINGSIDE, QUEENSIDE })
                if (castling_rights(side, ~turn()) && move.to() == get_rook_square(m_castling_rights[side][~turn()], ~turn()))
                    result ^= Zobrist::get_castle_side_turn(side, ~turn());
    }
    else if (move.is_double_pawn_push())
    {
        // Update ep square
        result ^= Zobrist::get_ep_file(file(move.to()));
    }
    else if (move.is_castle())
    {
        // Castling: update the hash of the rook
        CastleSide side = file(move.to()) >= 4 ? KINGSIDE : QUEENSIDE;
        Square iS = get_rook_square(m_castling_rights[side][turn()], turn());
        Square iE = move.to() + (side == KINGSIDE ? -1 : +1);
        result ^= Zobrist::get_piece_turn_square(ROOK, turn(), iS);
        result ^= Zobrist::get_piece_turn_square(ROOK, turn(), iE);
    }

    // Reset castling rights after a king or rook move
    if (piece == KING)
    {
        // Unset all castling rights after a king move
        for (auto side : { KINGSIDE, QUEENSIDE })
            if (castling_rights(side, turn()))
                result ^= Zobrist::get_castle_side_turn(side, turn());
    }
    else if (piece == ROOK)
    {
        // Unset castling rights for a certain side if a rook moves
        for (auto side : { KINGSIDE, QUEENSIDE })
            if (castling_rights(side, turn()) && move.from() == get_rook_square(m_castling_rights[side][turn()], turn()))
                result ^= Zobrist::get_castle_side_turn(side, turn());
    }

    // Reset previous en-passant hash
    if (m_enpassant_square != SQUARE_NULL)
        result ^= Zobrist::get_ep_file(file(m_enpassant_square));

    return result;
}


bool Board::is_valid() const
{
    // Side not to move in check?
    Square king_square = m_pieces[KING][~m_turn].bitscan_forward();
    if (attackers(king_square, get_pieces(), m_turn))
        return false;

    // Bitboard consistency
    Bitboard occupancy;
    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (auto turn : { WHITE, BLACK })
        {
            if (m_pieces[piece][turn] & occupancy)
                return false;
            occupancy |= m_pieces[piece][turn];
        }

    // Piece-square consistency
    for (Square square = 0; square < NUM_SQUARES; square++)
        if (m_board_pieces[square] == NO_PIECE)
        {
            if (occupancy.test(square))
                return false;
        }
        else if (!m_pieces[get_piece_at(square)][get_turn(m_board_pieces[square])].test(square))
            return false;

    // Hash consistency
    if (m_hash != generate_hash())
        return false;

    // Material and phase evaluation
    NNUE::Accumulator acc[2];
    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (Turn turn : { WHITE, BLACK })
        {
            Bitboard bb = get_pieces(turn, piece);
            while (bb)
            {
                Square s = bb.bitscan_forward_reset();
                acc[WHITE].push(piece, s, m_king_sq[WHITE], turn, WHITE);
                acc[BLACK].push(piece, s, m_king_sq[BLACK], turn, BLACK);
            }
        }
    if (acc[WHITE] != m_acc[WHITE] || acc[BLACK] != m_acc[BLACK])
        return false;

    return true;
}


Board Board::make_null_move()
{
    Board result = *this;

    // En-passant
    result.m_enpassant_square = SQUARE_NULL;
    if (m_enpassant_square != SQUARE_NULL)
        result.m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    // Swap turns
    result.m_turn = ~m_turn;
    result.m_hash ^= Zobrist::get_black_move();
    return result;
}


int Board::half_move_clock() const
{
    return m_half_move_clock;
}


Bitboard Board::get_pieces() const
{
    return get_pieces<WHITE>() | get_pieces<BLACK>();
}


Bitboard Board::get_pieces(Turn turn, PieceType piece) const
{
    return m_pieces[piece][turn];
}


Turn Board::turn() const
{
    return m_turn;
}


Bitboard Board::checkers() const
{
    return m_checkers;
}


Bitboard Board::attackers(Square square, Bitboard occupancy, Turn turn) const
{
    if (turn == WHITE)
        return attackers<WHITE>(square, occupancy);
    else
        return attackers<BLACK>(square, occupancy);
}


bool Board::operator==(const Board& other) const
{
    if (m_hash != other.m_hash)
        return false;

    if (m_turn != other.m_turn || m_enpassant_square != other.m_enpassant_square)
        return false;

    for (CastleSide side : { KINGSIDE, QUEENSIDE })
        for (Turn turn : { WHITE, BLACK })
            if (m_castling_rights[side][turn] != other.m_castling_rights[side][turn])
                return false;

    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (Turn turn : { WHITE, BLACK })
            if (!(m_pieces[piece][turn] == other.m_pieces[piece][turn]))
                return false;

    return true;
}


Hash Board::hash() const
{
    return m_hash;
}


Square Board::least_valuable(Bitboard bb) const
{
    // Return the least valuable piece in the bitboard
    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
    {
        Bitboard piece_bb = (get_pieces(WHITE, piece) | get_pieces(BLACK, piece)) & bb;
        if (piece_bb)
            return piece_bb.bitscan_forward();
    }

    return SQUARE_NULL;
}


Score Board::see(Move move, Score threshold) const
{
    // Static-Exchange evaluation with pruning
    Square target = move.to();

    // Make the initial capture
    PieceType last_attacker = get_piece_at(move.from());
    Score gain = piece_value[move.is_ep_capture() ? PAWN : get_piece_at(target)] - threshold;
    Bitboard from_bb = Bitboard::from_square(move.from());
    Bitboard occupancy = get_pieces() ^ from_bb;
    Turn side_to_move = ~m_turn;
    int color = -1;

    // Iterate over opponent attackers
    Bitboard attacks_target = attackers(target, occupancy, side_to_move) & occupancy;
    while (attacks_target)
    {
        // If the side to move is already ahead they can stop the capture sequence,
        // so we can prune the remaining iterations
        if (color * gain > 0)
            return gain;

        // Get least valuable attacker
        Square attacker = least_valuable(attacks_target);
        Bitboard attacker_bb = Bitboard::from_square(attacker);

        // Make the capture
        gain += color * piece_value[last_attacker];
        last_attacker = get_piece_at(attacker);
        occupancy ^= attacker_bb;
        side_to_move = ~side_to_move;
        color = -color;

        // Get opponent attackers
        attacks_target = attackers(target, occupancy, side_to_move) & occupancy;
    }

    return gain;
}


bool Board::legal(Move move) const
{
    // Same source and destination squares?
    if (move.from() == move.to() && !(UCI::Options::UCI_Chess960 && move.is_castle()))
        return false;

    // Ep without the square defined?
    if (move.is_ep_capture() && (m_enpassant_square == SQUARE_NULL || move.to() != m_enpassant_square))
        return false;

    // Valid movetype?
    if (move.move_type() == INVALID_1 || move.move_type() == INVALID_2)
        return false;

    // Source square is not ours?
    Bitboard our_pieces = (m_turn == WHITE ? get_pieces<WHITE>() : get_pieces<BLACK>());
    if (!our_pieces.test(move.from()))
        return false;

    // Destination square ours?
    if (our_pieces.test(move.to()) && !(UCI::Options::UCI_Chess960 && move.is_castle()))
        return false;

    // Capture and destination square not occupied by the opponent (including ep)?
    PieceType piece = get_piece_at(move.from());
    Bitboard enemy_pieces = get_pieces() & ~our_pieces;
    if (move.is_ep_capture() && piece == PAWN && m_enpassant_square != SQUARE_NULL)
        enemy_pieces.set(m_enpassant_square);
    if (enemy_pieces.test(move.to()) != move.is_capture())
        return false;

    // Pawn flags
    if (piece != PAWN && (move.is_double_pawn_push() ||
        move.is_ep_capture() ||
        move.is_promotion()))
        return false;

    // King flags
    if (piece != KING && move.is_castle())
        return false;

    Bitboard occupancy = get_pieces();
    if (piece == PAWN)
        return legal<PAWN>(move, occupancy);
    else if (piece == KNIGHT)
        return legal<KNIGHT>(move, occupancy);
    else if (piece == BISHOP)
        return legal<BISHOP>(move, occupancy);
    else if (piece == ROOK)
        return legal<ROOK>(move, occupancy);
    else if (piece == QUEEN)
        return legal<QUEEN>(move, occupancy);
    else if (piece == KING)
        return legal<KING>(move, occupancy);
    else
        return false;
}


Bitboard Board::non_pawn_material() const
{
    return get_pieces<WHITE, KNIGHT>() | get_pieces<BLACK, KNIGHT>()
         | get_pieces<WHITE, BISHOP>() | get_pieces<BLACK, BISHOP>()
         | get_pieces<WHITE, ROOK>()   | get_pieces<BLACK, ROOK>()
         | get_pieces<WHITE, QUEEN>()  | get_pieces<BLACK, QUEEN>();
}

Bitboard Board::non_pawn_material(Turn turn) const
{
    if (turn == WHITE)
        return get_pieces<WHITE, KNIGHT>() | get_pieces<WHITE, BISHOP>()
             | get_pieces<WHITE, ROOK  >() | get_pieces<WHITE, QUEEN>();
    else
        return get_pieces<BLACK, KNIGHT>() | get_pieces<BLACK, BISHOP>()
             | get_pieces<BLACK, ROOK  >() | get_pieces<BLACK, QUEEN>();
}


Bitboard Board::sliders() const
{
    return get_pieces<WHITE, BISHOP>() | get_pieces<BLACK, BISHOP>()
         | get_pieces<WHITE, ROOK>()   | get_pieces<BLACK, ROOK>()
         | get_pieces<WHITE, QUEEN>()  | get_pieces<BLACK, QUEEN>();
}



const NNUE::Accumulator& Board::accumulator(Turn t) const
{
    return m_acc[t];
}



std::string Board::to_uci(Move m) const
{
    // Castling for Chess960 (this needs to come before null moves)
    if (UCI::Options::UCI_Chess960 && m.is_castle())
    {
        CastleSide side = file(m.to()) >= 4 ? KINGSIDE : QUEENSIDE;
        Turn turn = rank(m.from()) < 4 ? WHITE : BLACK;
        return get_square(m.from()) + get_square(get_rook_square(m_castling_rights[side][turn], turn));
    }

    // Null moves
    if (m.from() == m.to())
        return "0000";

    // Promotions
    if (m.is_promotion())
    {
        // Promotion
        PieceType piece = m.promo_piece();
        char promo_code = piece == KNIGHT ? 'n'
                        : piece == BISHOP ? 'b'
                        : piece == ROOK   ? 'r'
                        :                   'q';
        return get_square(m.from()) + get_square(m.to()) + promo_code;
    }

    // Regular move
    return get_square(m.from()) + get_square(m.to());
}





Position::Position()
    : m_boards(1), m_stack(NUM_MAX_DEPTH), m_pos(0), m_moves(0)
{}


Position::Position(std::string fen)
    : Position()
{
    m_boards[0] = Board(fen);
}


bool Position::is_draw(bool unique) const
{
    // Fifty move rule
    if (board().half_move_clock() >= 100)
        return true;

    // Repetitions
    int cur_pos = (int)m_boards.size() - 1;
    int n_moves = std::min(cur_pos + 1, board().half_move_clock());
    int min_pos = cur_pos - n_moves + 1;
    if (n_moves >= (unique ? 4 : 8))
    {
        int pos1 = cur_pos - 4;
        while (pos1 >= min_pos)
        {
            if (board() == m_boards[pos1])
            {
                if (unique)
                    return true;
                int pos2 = pos1 - 4;
                while (pos2 >= min_pos)
                {
                    if (board() == m_boards[pos2])
                        return true;
                    pos2 -= 2;
                }

            }
            pos1 -= 2;
        }
    }

    return false;
}


bool Position::in_check() const
{
    return board().checkers();
}


Turn Position::get_turn() const
{
    return board().turn();
}


MoveList Position::generate_moves(MoveGenType type)
{
    auto list = m_stack.list();
    board().generate_moves(list, type);
    return list;
}


void Position::make_move(Move move)
{
    ++m_stack;
    ++m_pos;
    m_boards.push_back(board().make_move(move));
    m_moves.push_back(move);
}


void Position::unmake_move()
{
    m_boards.pop_back();
    m_moves.pop_back();
    --m_stack;
    --m_pos;
}


void Position::make_null_move()
{
    ++m_stack;
    ++m_pos;
    m_boards.push_back(board().make_null_move());
    m_moves.push_back(MOVE_NULL);
}


void Position::unmake_null_move()
{
    m_boards.pop_back();
    m_moves.pop_back();
    --m_stack;
    --m_pos;
}


Board& Position::board()
{
    return m_boards.back();
}


const Board& Position::board() const
{
    return m_boards.back();
}


Hash Position::hash() const
{
    return board().hash();
}


MoveList Position::move_list() const
{
    return m_stack.list();
}


void Position::set_init_ply()
{
    m_pos = 0;
    m_stack.reset_pos();
    std::size_t new_size = 256 * (m_boards.size() / 256 + 1);
    m_boards.reserve(new_size);
    m_moves.reserve(new_size);
}


Depth Position::ply() const
{
    return m_pos;
}


Move Position::last_move(std::size_t offset) const
{
    return m_moves.size() > offset ? m_moves[m_moves.size() - offset - 1] : MOVE_NULL;
}


const Board& Position::last_board() const
{
    return m_boards[m_boards.size() - 2];
}


std::ostream& operator<<(std::ostream& out, const Board& board)
{
    out << "   +------------------------+\n";
    for (int rank = 7; rank >= 0; rank--)
    {
        out << " " << rank + 1 << " |";
        for (int file = 0; file < 8; file++)
        {
            out << " ";
            Piece pc = board.m_board_pieces[make_square(rank, file)];
            if (pc == Piece::NO_PIECE)
                out << '.';
            else
                out << fen_piece(pc);
            out << " ";
        }
        out << "|\n";
        if (rank > 0)
            out << "   |                        |\n";
    }
    out << "   +------------------------+\n";
    out << "     A  B  C  D  E  F  G  H \n";

    out << "\n";
    out << "FEN: " << board.to_fen() << "\n";
    out << "Hash: " << std::hex << board.m_hash << std::dec << "\n";

    if (board.get_pieces().count() <= Syzygy::Cardinality)
    {
        auto probe_result = Syzygy::probe_dtz(board);
        Syzygy::WDL wdl = probe_result.first;
        int dtz = probe_result.second;
        if (wdl != Syzygy::WDL_NONE)
        {
            wdl = board.turn() == WHITE ? wdl : -wdl;
            out << "Tablebase: "
                << (wdl == Syzygy::WDL_LOSS         ? "Black wins"
                  : wdl == Syzygy::WDL_BLESSED_LOSS ? "Draw (cursed black win)"
                  : wdl == Syzygy::WDL_DRAW         ? "Draw"
                  : wdl == Syzygy::WDL_CURSED_WIN   ? "Draw (cursed white win)"
                  : wdl == Syzygy::WDL_WIN          ? "White wins"
                  :                                   "error")
                << " - DTZ: " << dtz
                << "\n";
        }
    }
    return out;
}
