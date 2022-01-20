#include "Position.hpp"
#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include "Zobrist.hpp"
#include <cassert>
#include <sstream>
#include <string>
#include <cctype>
#include <cstring>


Piece fen_piece(char c)
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


char fen_piece(BoardPieces pc)
{
    Piece piece = get_piece(pc);
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
      m_eval1(0, 0),
      m_phase(Phases::Total)
{
    char c, c2;
    std::istringstream ss(fen);

    // Default initialisation for updated fields
    std::memset(m_board_pieces, 0, sizeof(m_board_pieces));

    // Read position
    Square square = SQUARE_A8;
    while (ss.get(c) && !isspace(c))
    {
        if (isdigit(c))
            square += c - '0';
        else if (c == '/')
            square -= 16;
        else
            set_piece<true>(fen_piece(c), isupper(c) ? WHITE : BLACK, square++);
    }

    // Side to move
    while (ss.get(c) && !isspace(c))
        m_turn = (c == 'w') ? WHITE : BLACK;

    // Castling rights
    std::memset(m_castling_rights, 0, sizeof(m_castling_rights));
    while (ss.get(c) && !isspace(c))
        if (fen_castle_side(c) != NO_SIDE)
            set_castling<true>(fen_castle_side(c), isupper(c) ? WHITE : BLACK);

    // Ep square
    m_enpassant_square = SQUARE_NULL;
    while (ss.get(c) && !isspace(c))
        if (c != '-' && ss.get(c2) && !isspace(c2))
            m_enpassant_square = make_square(c2 - '1', c - 'a');

    // Half-move clock
    m_half_move_clock = 0;
    if (ss)
        ss >> m_half_move_clock;

    // Full-move clock
    m_full_move_clock = 0;
    if (ss)
        ss >> m_full_move_clock;

    // Update remaining hash: turn and ep square
    if (m_turn == Turn::BLACK)
        m_hash ^= Zobrist::get_black_move();
    if (m_enpassant_square != SQUARE_NULL)
        m_hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    update_checkers();
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
            BoardPieces pc = m_board_pieces[make_square(rank, file)];
            if (pc == BoardPieces::NO_PIECE)
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
    for (int turn = 0; turn < 2; turn++)
        for (int piece = 0; piece < 6; piece++)
        {
            Bitboard piece_bb = m_pieces[piece][turn];
            while (piece_bb)
                hash ^= Zobrist::get_piece_turn_square(static_cast<PieceType>(piece),
                    static_cast<Turn>(turn),
                    piece_bb.bitscan_forward_reset());
        }

    // Turn to move
    if (m_turn == Turn::BLACK)
        hash ^= Zobrist::get_black_move();

    // En-passsant square
    if (m_enpassant_square != SQUARE_NULL)
        hash ^= Zobrist::get_ep_file(file(m_enpassant_square));

    // Castling rights
    for (int side = 0; side < 2; side++)
        for (int turn = 0; turn < 2; turn++)
            if (m_castling_rights[side][turn])
                hash ^= Zobrist::get_castle_side_turn(static_cast<CastleSide>(side), static_cast<Turn>(turn));

    return hash;
}


void Board::update_checkers()
{
    if (m_turn == WHITE)
        update_checkers<WHITE>();
    else
        update_checkers<BLACK>();
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
    Direction up = (m_turn == WHITE) ? 8 : -8;
    PieceType piece = static_cast<PieceType>(get_piece_at(move.from()));

    // Increment clocks
    result.m_full_move_clock += m_turn;
    if (piece == PAWN || move.is_capture())
        result.m_half_move_clock = 0;
    else
        result.m_half_move_clock++;

    // Initial empty ep square
    result.m_enpassant_square = SQUARE_NULL;

    // Castling rights
    if (piece == KING)
    {
        for (auto side : { KINGSIDE, QUEENSIDE })
            result.set_castling<false>(side, m_turn);
    }
    else if (piece == ROOK)
    {
        // Initial rook positions
        if (move.from() == (m_turn == WHITE ? SQUARE_H1 : SQUARE_H8))
            result.set_castling<false>(KINGSIDE, m_turn);
        if (move.from() == (m_turn == WHITE ? SQUARE_A1 : SQUARE_A8))
            result.set_castling<false>(QUEENSIDE, m_turn);
    }

    // Remove moving piece
    if (move.is_promotion())
        result.pop_piece<true>(piece, m_turn, move.from());
    else
        result.pop_piece<false>(piece, m_turn, move.from());

    // Per move type action
    if (move.is_capture())
    {
        Square target = move.to();
        PieceType target_piece = get_piece_at(move.to());

        // En passant capture: update target square and piece
        if (move.is_ep_capture())
        {
            target = move.to() - up;
            target_piece = PAWN;
        }

        // Remove captured piece
        result.pop_piece<true>(target_piece, ~m_turn, target);

        // Castling: check if any rook has been captured
        if (move.to() == (m_turn == WHITE ? SQUARE_H8 : SQUARE_H1))
            result.set_castling<false>(KINGSIDE, ~m_turn);
        if (move.to() == (m_turn == WHITE ? SQUARE_A8 : SQUARE_A1))
            result.set_castling<false>(QUEENSIDE, ~m_turn);
    }
    else if (move.is_double_pawn_push())
    {
        result.m_enpassant_square = move.to() - up;
        result.m_hash ^= Zobrist::get_ep_file(file(move.to()));
    }
    else if (move.is_castle())
    {
        // Move the rook to the new square
        Square iS = move.to() + (move.to() > move.from() ? +1 : -2);
        Square iE = move.to() + (move.to() > move.from() ? -1 : +1);
        result.pop_piece<false>(ROOK, m_turn, iS);
        result.set_piece<false>(ROOK, m_turn, iE);
    }

     // Set moving piece
    if (move.is_promotion())
    {
        PieceType promo_piece = static_cast<PieceType>(move.promo_piece());
        result.set_piece<true>(promo_piece, m_turn, move.to());
    }
    else
        result.set_piece<false>(piece, m_turn, move.to());

    // Swap turns
    result.m_turn = ~m_turn;
    result.m_hash ^= Zobrist::get_black_move();

    // If en-passant was played reset the bit
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
    int8_t phase = Phases::Total;
    auto eval = MixedScore(0, 0);
    for (Piece piece = PAWN; piece < NUM_PIECE_TYPES; piece++)
        for (Turn turn : { WHITE, BLACK })
        {
            Bitboard bb = get_pieces(turn, piece);
            eval += piece_value[piece] * bb.count() * turn_to_color(turn);
            phase -= bb.count() * Phases::Pieces[piece];
            while (bb)
                eval += piece_square(piece, bb.bitscan_forward_reset(), turn) * turn_to_color(turn);
        }
    if (phase != m_phase)
        return false;
    if (eval.middlegame() != m_eval1.middlegame() || eval.endgame() != m_eval1.endgame())
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


Bitboard Board::get_pieces(Turn turn, Piece piece) const
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

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (m_castling_rights[i][j] != other.m_castling_rights[i][j])
                return false;

    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 2; j++)
            if (!(m_pieces[i][j] == other.m_pieces[i][j]))
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
    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
    {
        Bitboard piece_bb = (get_pieces(WHITE, piece) | get_pieces(BLACK, piece)) & bb;
        if (piece_bb)
            return piece_bb.bitscan_forward();
    }

    return SQUARE_NULL;
}


Score Board::see(Move move, int threshold) const
{
    // Static-Exchange evaluation with pruning
    constexpr int piece_score[] = { 0, 100, 300, 300, 500, 900, 10000 };

    Square target = move.to();

    // Make the initial capture
    Piece last_attacker = get_piece_at(move.from());
    int gain = piece_score[1 + (move.is_ep_capture() ? PAWN : get_piece_at(target))] - threshold;
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
        gain += color * piece_score[1 + last_attacker];
        last_attacker = get_piece_at(attacker);
        occupancy ^= attacker_bb;
        side_to_move = ~side_to_move;
        color = -color;

        // Get opponent attackers
        attacks_target = attackers(target, occupancy, side_to_move) & occupancy;
    }

    return gain;
}


MixedScore Board::material_eval() const
{
    return m_eval1;
}


int8_t Board::phase() const
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
    Piece piece = get_piece_at(move.from());
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
    : m_boards(1)
{
    m_boards.reserve(NUM_BOARDS);
}


Position::Position(std::string fen)
    : Position()
{
    m_boards[0] = Board(fen);
}


void Position::reset_startpos()
{
    m_boards.clear();
    m_boards.push_back(Board());
}


void Position::update_from(std::string fen)
{
    m_boards.clear();
    m_boards.push_back(Board(fen));
}


void Position::update_from(const Position& pos)
{
    m_boards.clear();
    // Resize if needed
    if (m_boards.capacity() < pos.m_boards.capacity())
        m_boards.reserve(pos.m_boards.capacity());
    // Push moves
    for (const auto& board : pos.m_boards)
        m_boards.push_back(board);
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


bool Position::is_check() const
{
    return board().checkers();
}


Turn Position::get_turn() const
{
    return board().turn();
}


void Position::make_move(Move move)
{
    m_boards.push_back(board().make_move(move));
}


void Position::unmake_move()
{
    m_boards.pop_back();
}


void Position::make_null_move()
{
    m_boards.push_back(board().make_null_move());
}


void Position::unmake_null_move()
{
    m_boards.pop_back();
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

MoveList Position::generate_moves(MoveGenType type) const
{
    MoveList list;
    board().generate_moves(list, type);
    return list;
}

void Position::prepare()
{
    if (m_boards.capacity() < m_boards.size() + NUM_MAX_PLY)
        m_boards.reserve(m_boards.size() + 2 * NUM_MAX_PLY);
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
            BoardPieces pc = board.m_board_pieces[make_square(rank, file)];
            if (pc == BoardPieces::NO_PIECE)
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
