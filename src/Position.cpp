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
    m_psq[turn].clear();
    for (PieceType p : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
    {
        for (Turn t : { WHITE, BLACK })
        {
            Bitboard b = get_pieces(t, p);
            while (b)
                m_psq[turn].push(p, b.bitscan_forward_reset(), m_king_sq[turn], t, turn);
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

    // Update castling rights after this move
    if (piece == KING)
    {
        // Unset all castling rights after a king move
        for (auto side : { KINGSIDE, QUEENSIDE })
            result.set_castling<false>(side, m_turn);
    }
    else if (piece == ROOK)
    {
        // Unset castling rights for a certain side if a rook moves
        if (move.from() == (m_turn == WHITE ? SQUARE_H1 : SQUARE_H8))
            result.set_castling<false>(KINGSIDE, m_turn);
        if (move.from() == (m_turn == WHITE ? SQUARE_A1 : SQUARE_A8))
            result.set_castling<false>(QUEENSIDE, m_turn);
    }

    // Per move type action
    if (move.is_capture())
    {
        // Captured square is different for ep captures
        Square target = move.is_ep_capture() ? move.to() - up : move.to();

        // Remove captured piece
        result.pop_piece(get_piece_at(target), ~m_turn, target);

        // Castling: check if any rook has been captured
        if (move.to() == (m_turn == WHITE ? SQUARE_H8 : SQUARE_H1))
            result.set_castling<false>(KINGSIDE, ~m_turn);
        if (move.to() == (m_turn == WHITE ? SQUARE_A8 : SQUARE_A1))
            result.set_castling<false>(QUEENSIDE, ~m_turn);
    }
    else if (move.is_double_pawn_push())
    {
        // Update ep square
        result.m_enpassant_square = move.to() - up;
        result.m_hash ^= Zobrist::get_ep_file(file(move.to()));
    }
    else if (move.is_castle())
    {
        // Move the rook to the new square
        Square iS = move.to() + (move.to() > move.from() ? +1 : -2);
        Square iE = move.to() + (move.to() > move.from() ? -1 : +1);
        result.move_piece(ROOK, m_turn, iS, iE);
    }

    // Set piece on target square
    if (move.is_promotion())
    {
        result.pop_piece(piece, m_turn, move.from());
        result.set_piece(move.promo_piece(), m_turn, move.to());
    }
    else
    {
        result.move_piece(piece, m_turn, move.from(), move.to());
    }

    // After a king move, update PSQ tables
    if (piece == KING)
    {
        result.m_king_sq[m_turn] = result.m_pieces[KING][m_turn].bitscan_forward();
        result.regen_psqt(m_turn);
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
    PSQT::Accumulator acc[2];
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
                acc[WHITE].push(piece, s, m_king_sq[WHITE], turn, WHITE);
                acc[BLACK].push(piece, s, m_king_sq[BLACK], turn, BLACK);
            }
        }
    MixedScore total_material = m_material[WHITE] - m_material[BLACK];
    if (phase != m_phase)
        return false;
    if (material.middlegame() != total_material.middlegame() || material.endgame() != total_material.endgame())
        return false;
    if (acc[WHITE] != m_psq[WHITE] || acc[BLACK] != m_psq[BLACK])
        return false;
    if (material_hash != m_material_hash)
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
    return m_psq[WHITE].eval() - m_psq[BLACK].eval();
}


MixedScore Board::psq(Turn t) const
{
    return m_psq[t].eval();
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
    : m_boards(1), m_stack(NUM_MAX_DEPTH), m_pos(0), m_extensions(0), m_moves(0), m_reduced(false)
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
    if (n_moves >= 8)
    {
        int pos1 = cur_pos - 4;
        while (pos1 >= min_pos)
        {
            if (board().hash() == m_boards[pos1].hash())
            {
                if (unique)
                    return true;
                int pos2 = pos1 - 4;
                while (pos2 >= min_pos)
                {
                    if (board().hash() == m_boards[pos2].hash())
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


void Position::make_move(Move move, bool extension)
{
    ++m_stack;
    ++m_pos;
    m_boards.push_back(board().make_move(move));
    m_moves.push_back(MoveInfo{ move, extension });

    if (extension)
        m_extensions++;
}


void Position::unmake_move()
{
    m_boards.pop_back();
    --m_stack;
    --m_pos;

    auto info = m_moves.back();
    if (info.extended)
        m_extensions--;

    m_moves.pop_back();
}


void Position::make_null_move()
{
    ++m_stack;
    ++m_pos;
    m_boards.push_back(board().make_null_move());
    m_moves.push_back(MoveInfo{ MOVE_NULL, false });
}


void Position::unmake_null_move()
{
    m_boards.pop_back();
    --m_stack;
    --m_pos;
    m_moves.pop_back();
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


int Position::num_extensions() const
{
    return m_extensions;
}


void Position::set_init_ply()
{
    m_pos = 0;
    m_stack.reset_pos();
}


Depth Position::ply() const
{
    return m_pos;
}


bool Position::reduced() const
{
    return m_reduced;
}


Move Position::last_move(std::size_t offset) const
{
    return m_moves.size() > offset ? m_moves[m_moves.size() - offset - 1].move : MOVE_NULL;
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
