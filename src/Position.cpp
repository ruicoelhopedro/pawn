#include "Position.hpp"
#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include "Zobrist.hpp"
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


char fen_castle_side(CastleSide side, Turn turn)
{
    char c = side == KINGSIDE  ? 'k'
           : side == QUEENSIDE ? 'q'
           : 'x';
    return turn == WHITE ? toupper(c) : c;
}


Board::Board()
    : Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
{}


Board::Board(std::string fen)
    : m_hash(0),
      m_material_hash(Zobrist::get_initial_material_hash()),
      m_phase(Phases::Total)
{
    auto c = fen.cbegin();

    // Default initialisation for board pieces
    std::memset(m_board_pieces, PIECE_NONE, sizeof(m_board_pieces));
    std::memset(m_piece_count, 0, sizeof(m_piece_count));
    std::memset(m_psq, 0, sizeof(m_psq));
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
            set_piece<true>(fen_piece(*c), isupper(*c) ? WHITE : BLACK, square++);

        c++;
    }

    // Side to move
    m_turn = WHITE;
    while ((++c) < fen.cend() && !isspace(*c))
        m_turn = (*c == 'w') ? WHITE : BLACK;

    // Castling rights
    std::memset(m_castling_rights, 0, sizeof(m_castling_rights));
    while ((++c) < fen.cend() && !isspace(*c))
        if (fen_castle_side(*c) != NO_SIDE)
            set_castling<true>(fen_castle_side(*c), isupper(*c) ? WHITE : BLACK);

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

    m_king_sq[WHITE] = m_pieces[KING][WHITE].bitscan_forward();
    m_king_sq[BLACK] = m_pieces[KING][BLACK].bitscan_forward();
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
            if (m_castling_rights[side][turn])
            {
                found = true;
                ss << fen_castle_side(side, turn);
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
            if (m_castling_rights[side][turn])
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
    m_psq[turn] = MixedScore(0, 0);
    for (PieceType p : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
    {
        for (Turn t : { WHITE, BLACK })
        {
            Bitboard b = get_pieces(t, p);
            while (b)
            {
                Square s = b.bitscan_forward_reset();
                m_psq[turn] += piece_square(p, s, t, m_king_sq[turn], turn) * turn_to_color(t);
            }
        }
    }
}


void Board::generate_moves(MoveList& list, MoveGenType type) const
{
    if (m_turn == WHITE)
        generate_moves<WHITE>(list, type);
    else
        generate_moves<BLACK>(list, type);
}


MoveInfo Board::make_move(Move move)
{
    const Direction up = (m_turn == WHITE) ? 8 : -8;
    const PieceType piece = get_piece_at(move.from());

    // Store move info
    MoveInfo mi;
    mi.move = move;
    mi.captured = PIECE_NONE;
    mi.ep_square = m_enpassant_square;
    mi.half_move_clock = m_half_move_clock;
    std::memcpy(mi.castling_rights, m_castling_rights, sizeof(m_castling_rights));
    mi.hash = m_hash;
    std::memcpy(mi.psq, m_psq, sizeof(m_psq));

    // Increment clocks
    m_full_move_clock += m_turn;
    if (piece == PAWN || move.is_capture())
        m_half_move_clock = 0;
    else
        m_half_move_clock++;

    // Reset previous en-passant hash
    if (m_enpassant_square != SQUARE_NULL)
        m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    // Initial empty ep square
    m_enpassant_square = SQUARE_NULL;

    // Update castling rights after this move
    if (piece == KING)
    {
        // Unset all castling rights after a king move
        for (auto side : { KINGSIDE, QUEENSIDE })
            set_castling<false>(side, m_turn);
    }
    else if (piece == ROOK)
    {
        // Unset castling rights for a certain side if a rook moves
        if (move.from() == (m_turn == WHITE ? SQUARE_H1 : SQUARE_H8))
            set_castling<false>(KINGSIDE, m_turn);
        if (move.from() == (m_turn == WHITE ? SQUARE_A1 : SQUARE_A8))
            set_castling<false>(QUEENSIDE, m_turn);
    }

    // Per move type action
    if (move.is_capture())
    {
        // Captured square is different for ep captures
        Square target = move.is_ep_capture() ? move.to() - up : move.to();

        // Remove captured piece
        PieceType captured = get_piece_at(target);
        pop_piece<true>(captured, ~m_turn, target);
        mi.captured = captured;

        // Castling: check if any rook has been captured
        if (move.to() == (m_turn == WHITE ? SQUARE_H8 : SQUARE_H1))
            set_castling<false>(KINGSIDE, ~m_turn);
        if (move.to() == (m_turn == WHITE ? SQUARE_A8 : SQUARE_A1))
            set_castling<false>(QUEENSIDE, ~m_turn);
    }
    else if (move.is_double_pawn_push())
    {
        // Update ep square
        m_enpassant_square = move.to() - up;
        m_hash ^= Zobrist::get_ep_file(file(move.to()));
    }
    else if (move.is_castle())
    {
        // Move the rook to the new square
        Square iS = move.to() + (move.to() > move.from() ? +1 : -2);
        Square iE = move.to() + (move.to() > move.from() ? -1 : +1);
        move_piece<true>(ROOK, m_turn, iS, iE);
    }

    // Set piece on target square
    if (move.is_promotion())
    {
        pop_piece<true>(piece, m_turn, move.from());
        set_piece<true>(move.promo_piece(), m_turn, move.to());
    }
    else
    {
        move_piece<true>(piece, m_turn, move.from(), move.to());
    }

    // After a king move, update PSQ tables
    if (piece == KING)
    {
        m_king_sq[m_turn] = m_pieces[KING][m_turn].bitscan_forward();
        regen_psqt(m_turn);
    }

    // Swap turns
    m_turn = ~m_turn;
    m_hash ^= Zobrist::get_black_move();

    // Update checkers
    update_checkers();

    return mi;
}


MoveInfo Board::make_null_move()
{
    // Store move info
    MoveInfo mi;
    mi.move = MOVE_NULL;
    mi.captured = PIECE_NONE;
    mi.ep_square = m_enpassant_square;
    mi.half_move_clock = m_half_move_clock;
    std::memcpy(mi.castling_rights, m_castling_rights, sizeof(m_castling_rights));
    mi.hash = m_hash;
    std::memcpy(mi.psq, m_psq, sizeof(m_psq));

    // En-passant
    if (m_enpassant_square != SQUARE_NULL)
        m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));
    m_enpassant_square = SQUARE_NULL;

    // Swap turns
    m_turn = ~m_turn;
    m_hash ^= Zobrist::get_black_move();

    return mi;
}


void Board::unmake_move(const MoveInfo& mi)
{
    // Restore board state
    m_enpassant_square = mi.ep_square;
    m_half_move_clock = mi.half_move_clock;
    std::memcpy(m_castling_rights, mi.castling_rights, sizeof(m_castling_rights));
    m_hash = mi.hash;
    std::memcpy(m_psq, mi.psq, sizeof(m_psq));

    // Swap turns
    m_turn = ~m_turn;
    const Direction up = (m_turn == WHITE) ? 8 : -8;

    // Restore moved piece
    PieceType piece = get_piece_at(mi.move.to());
    if (mi.move.is_promotion())
    {
        pop_piece<false>(mi.move.promo_piece(), m_turn, mi.move.to());
        set_piece<false>(PAWN, m_turn, mi.move.from());
    }
    else
    {
        move_piece<false>(piece, m_turn, mi.move.to(), mi.move.from());
    }

    // Special case: castling
    if (mi.move.is_castle())
    {
        // Move the rook to the new square
        Square iS = mi.move.to() + (mi.move.to() > mi.move.from() ? +1 : -2);
        Square iE = mi.move.to() + (mi.move.to() > mi.move.from() ? -1 : +1);
        move_piece<false>(ROOK, m_turn, iE, iS);
    }

    // Restore captured piece, if any
    if (mi.move.is_capture())
    {
        Square target = mi.move.is_ep_capture() ? mi.move.to() - up : mi.move.to();
        set_piece<false>(mi.captured, ~m_turn, target);
    }

    // After a king move, update king square
    if (piece == KING)
        m_king_sq[m_turn] = m_pieces[KING][m_turn].bitscan_forward();

    // Update checkers
    update_checkers();
}


void Board::unmake_null_move(const MoveInfo& mi)
{
    // Restore board state
    m_enpassant_square = mi.ep_square;
    m_half_move_clock = mi.half_move_clock;
    std::memcpy(m_castling_rights, mi.castling_rights, sizeof(m_castling_rights));
    m_hash = mi.hash;
    std::memcpy(m_psq, mi.psq, sizeof(m_psq));

    // Swap turns
    m_turn = ~m_turn;
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
    uint8_t phase = Phases::Total;
    MixedScore material(0, 0);
    MixedScore psq_sides[2] = { MixedScore(0, 0), MixedScore(0, 0) };
    Hash material_hash = 0;
    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (Turn turn : { WHITE, BLACK })
        {
            Bitboard bb = get_pieces(turn, piece);
            if (bb.count() != m_piece_count[piece][turn])
                return false;
            material += piece_value[piece] * bb.count() * turn_to_color(turn);
            phase -= bb.count() * Phases::Pieces[piece];
            material_hash ^= Zobrist::get_piece_turn_square(piece, turn, bb.count());
            while (bb)
            {
                Square s = bb.bitscan_forward_reset();
                psq_sides[WHITE] += piece_square(piece, s, turn, m_king_sq[WHITE], WHITE) * turn_to_color(turn);
                psq_sides[BLACK] += piece_square(piece, s, turn, m_king_sq[BLACK], BLACK) * turn_to_color(turn);
            }
        }
    MixedScore total_material = m_material[WHITE] - m_material[BLACK];
    if (phase != m_phase)
        return false;
    if (material.middlegame() != total_material.middlegame() || material.endgame() != total_material.endgame())
        return false;
    if (psq_sides[WHITE].middlegame() != m_psq[WHITE].middlegame() || psq_sides[BLACK].middlegame() != m_psq[BLACK].middlegame())
        return false;
    if (psq_sides[WHITE].endgame() != m_psq[WHITE].endgame() || psq_sides[BLACK].endgame() != m_psq[BLACK].endgame())
        return false;
    if (material_hash != m_material_hash)
        return false;

    return true;
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


Hash Board::material_hash() const
{
    return m_material_hash;
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
    Score gain = piece_value_mg[move.is_ep_capture() ? PAWN : get_piece_at(target)] - threshold;
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
        gain += color * piece_value_mg[last_attacker];
        last_attacker = get_piece_at(attacker);
        occupancy ^= attacker_bb;
        side_to_move = ~side_to_move;
        color = -color;

        // Get opponent attackers
        attacks_target = attackers(target, occupancy, side_to_move) & occupancy;
    }

    return gain;
}


MixedScore Board::material() const
{
    return m_material[WHITE] - m_material[BLACK];
}


MixedScore Board::material(Turn turn) const
{
    return m_material[turn] - KingValue;
}


MixedScore Board::psq() const
{
    return m_psq[WHITE] + m_psq[BLACK];
}


uint8_t Board::phase() const
{
    return m_phase;
}


bool Board::legal(Move move) const
{
    // Same source and destination squares?
    if (move.from() == move.to())
        return false;

    // Ep without the square defined?
    if (move.is_ep_capture() && (m_enpassant_square == SQUARE_NULL || move.to() != m_enpassant_square))
        return false;

    // Valid movetype?
    if (move.move_type() == INVALID_1 || move.move_type() == INVALID_2)
        return false;

    // Source square is not ours or destination ours?
    Bitboard our_pieces = (m_turn == WHITE ? get_pieces<WHITE>() : get_pieces<BLACK>());
    if (!our_pieces.test(move.from()) || our_pieces.test(move.to()))
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





Position::Position()
    : m_stack(NUM_MAX_DEPTH), m_ply(0)
{}


Position::Position(std::string fen)
    : Position()
{
    m_board = Board(fen);
}


bool Position::is_draw(bool two_fold) const
{
    // Fifty move rule
    if (board().half_move_clock() >= 100)
        return true;

    // Repetitions: we only need to consider the game history up to the move reseting the
    // halfmove clock, but limited to the number of plies we have
    int cur_pos = m_info.size();
    int n_moves = std::min(cur_pos + 1, board().half_move_clock());
    int min_pos = cur_pos - n_moves + 1;
    if (n_moves >= 8)
    {
        // Search for the first possible repetition. We use the position hashes for
        // comparing positions
        for (int pos1 = cur_pos - 4; pos1 >= min_pos; pos1 -= 2)
            if (board().hash() == m_info[pos1].hash)
            {
                // Two-fold repetition detected
                if (two_fold)
                    return true;

                // If we are looking for three-fold repetitions, keep searching for a second
                // repetition and return in that case
                for (int pos2 = pos1 - 4; pos2 >= min_pos; pos2 -= 2)
                    if (board().hash() == m_info[pos2].hash)
                        return true;
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
    m_info.push_back(m_board.make_move(move));
    ++m_ply;
    ++m_stack;
}


void Position::unmake_move()
{
    m_board.unmake_move(m_info.back());
    m_info.pop_back();
    --m_ply;
    --m_stack;
}


void Position::make_null_move()
{
    m_info.push_back(m_board.make_null_move());
    ++m_ply;
    ++m_stack;
}


void Position::unmake_null_move()
{
    m_board.unmake_null_move(m_info.back());
    m_info.pop_back();
    --m_ply;
    --m_stack;
}


Board& Position::board()
{
    return m_board;
}


const Board& Position::board() const
{
    return m_board;
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
    m_ply = 0;
    m_stack.reset_pos();
}


Depth Position::ply() const
{
    return m_ply;
}


Move Position::last_move() const
{
    return m_ply > 0 ? m_info.back().move : MOVE_NULL;
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
    return out;
}
