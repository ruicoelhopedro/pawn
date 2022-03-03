#include "Position.hpp"
#include "Types.hpp"
#include "PieceSquareTables.hpp"
#include "Zobrist.hpp"
#include <cassert>
#include <sstream>
#include <string>

Board::Board()
    : Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
{}


Board::Board(std::string fen)
    : m_eval1(0, 0)
{
    auto parts = split(fen, ' ');
    assert(parts.size() == 6 && "Invalid FEN, wrong number of fields");

    unsigned int pos = 0;
    for (int i = 7; i >= 0; i--)
    {
        int n_empty = 0;
        for (int j = 0; j < 8; j++)
        {
            if (n_empty > 1)
            {
                n_empty--;
                continue;
            }
            int index = i * 8 + j;
            assert(pos < parts[0].size() && "Invalid FEN, too short");
            char current = parts[0][pos];
            if (current == 'P')
                m_pieces[PAWN][WHITE].set(index);
            else if (current == 'p')
                m_pieces[PAWN][BLACK].set(index);
            else if (current == 'N')
                m_pieces[KNIGHT][WHITE].set(index);
            else if (current == 'n')
                m_pieces[KNIGHT][BLACK].set(index);
            else if (current == 'B')
                m_pieces[BISHOP][WHITE].set(index);
            else if (current == 'b')
                m_pieces[BISHOP][BLACK].set(index);
            else if (current == 'R')
                m_pieces[ROOK][WHITE].set(index);
            else if (current == 'r')
                m_pieces[ROOK][BLACK].set(index);
            else if (current == 'Q')
                m_pieces[QUEEN][WHITE].set(index);
            else if (current == 'q')
                m_pieces[QUEEN][BLACK].set(index);
            else if (current == 'K')
                m_pieces[KING][WHITE].set(index);
            else if (current == 'k')
                m_pieces[KING][BLACK].set(index);
            else if (current == '1')
                n_empty = 1;
            else if (current == '2')
                n_empty = 2;
            else if (current == '3')
                n_empty = 3;
            else if (current == '4')
                n_empty = 4;
            else if (current == '5')
                n_empty = 5;
            else if (current == '6')
                n_empty = 6;
            else if (current == '7')
                n_empty = 7;
            else if (current == '8')
                n_empty = 8;
            else
                assert(false && "Invalid FEN, unknown character");

            pos++;
        }
        if (i > 0)
        {
            assert(pos < parts[0].size() && "Invalid FEN, too short");
            assert(parts[0][pos] == '/' && "Invalid FEN, missing /");
            pos++;
        }
    }

    // Turn
    assert(parts[1].size() == 1 && "Invalid FEN, bad active color");
    if (parts[1][0] == 'w')
        m_turn = Turn::WHITE;
    else if (parts[1][0] == 'b')
        m_turn = Turn::BLACK;
    else
        assert(false && "Invalid FEN, unknown turn specifier");

    // Castling rights
    m_castling_rights[KINGSIDE][WHITE] = false;
    m_castling_rights[QUEENSIDE][WHITE] = false;
    m_castling_rights[KINGSIDE][BLACK] = false;
    m_castling_rights[QUEENSIDE][BLACK] = false;
    assert(parts[2].size() > 0 && parts[2].size() < 5 && "Invalid FEN, bad castling rights");
    if (parts[2][0] == '-')
    {
        assert(parts[2].size() == 1 && "Invalid FEN, bad castling rights");
    }
    else
    {
        for (unsigned int i = 0; i < parts[2].size(); i++)
        {
            if (parts[2][i] == 'K' && !m_castling_rights[KINGSIDE][WHITE])
                m_castling_rights[KINGSIDE][WHITE] = true;
            else if (parts[2][i] == 'Q' && !m_castling_rights[QUEENSIDE][WHITE])
                m_castling_rights[QUEENSIDE][WHITE] = true;
            else if (parts[2][i] == 'k' && !m_castling_rights[KINGSIDE][BLACK])
                m_castling_rights[KINGSIDE][BLACK] = true;
            else if (parts[2][i] == 'q' && !m_castling_rights[QUEENSIDE][BLACK])
                m_castling_rights[QUEENSIDE][BLACK] = true;
            else
                assert(false && "Invalid FEN, unknown castling right");
        }
    }

    // En passant
    assert(parts[3].size() > 0 && parts[3].size() < 3 && "Invalid FEN, bad castling rights");
    if (parts[3] == "-")
    {
        m_enpassant_square = -1;
    }
    else
    {
        int i;
        int j;
        if (parts[3][0] == 'a')
            j = 0;
        else if (parts[3][0] == 'b')
            j = 1;
        else if (parts[3][0] == 'c')
            j = 2;
        else if (parts[3][0] == 'd')
            j = 3;
        else if (parts[3][0] == 'e')
            j = 4;
        else if (parts[3][0] == 'f')
            j = 5;
        else if (parts[3][0] == 'g')
            j = 6;
        else if (parts[3][0] == 'h')
            j = 7;
        else
            assert(false && "Invalid FEN, unknown en passant square");

        if (parts[3][1] == '1')
            i = 0;
        else if (parts[3][1] == '2')
            i = 1;
        else if (parts[3][1] == '3')
            i = 2;
        else if (parts[3][1] == '4')
            i = 3;
        else if (parts[3][1] == '5')
            i = 4;
        else if (parts[3][1] == '6')
            i = 5;
        else if (parts[3][1] == '7')
            i = 6;
        else if (parts[3][1] == '8')
            i = 7;
        else
            assert(false && "Invalid FEN, unknown en passant square");

        m_enpassant_square = i * 8 + j;
    }

    // Half-move clock
    assert(parts[4].size() > 0 && "Invalid FEN, bad half-move clock");
    std::stringstream ss_half(parts[4]);
    ss_half >> m_half_move_clock;
    assert(!ss_half.fail() && "Invalid FEN, bad half-move clock");

    // Full-move clock
    assert(parts[5].size() > 0 && "Invalid FEN, bad full-move clock");
    std::stringstream ss_full(parts[5]);
    ss_full >> m_full_move_clock;
    assert(!ss_full.fail() && "Invalid FEN, bad full-move clock");

    init();
}


std::string Board::to_fen() const
{
    Bitboard white_pieces = m_pieces[0][0] | m_pieces[1][0] | m_pieces[2][0]
                          | m_pieces[3][0] | m_pieces[4][0] | m_pieces[5][0];
    Bitboard black_pieces = m_pieces[0][1] | m_pieces[1][1] | m_pieces[2][1]
                          | m_pieces[3][1] | m_pieces[4][1] | m_pieces[5][1];
    Bitboard occupied = white_pieces | black_pieces;

    std::string out;

    for (int i = 7; i >= 0; i--)
    {
        int n_empty = 0;
        for (int j = 0; j < 8; j++)
        {
            int index = i * 8 + j;
            if (!occupied.test(index))
            {
                n_empty++;
            }
            else
            {
                if (n_empty != 0)
                {
                    out += std::to_string(n_empty);
                    n_empty = 0;
                }
                if (white_pieces.test(index))
                {
                    if (m_pieces[0][0].test(index))
                        out += "P";
                    else if (m_pieces[1][0].test(index))
                        out += "N";
                    else if (m_pieces[2][0].test(index))
                        out += "B";
                    else if (m_pieces[3][0].test(index))
                        out += "R";
                    else if (m_pieces[4][0].test(index))
                        out += "Q";
                    else if (m_pieces[5][0].test(index))
                        out += "K";
                }
                else
                {
                    if (m_pieces[0][1].test(index))
                        out += "p";
                    else if (m_pieces[1][1].test(index))
                        out += "n";
                    else if (m_pieces[2][1].test(index))
                        out += "b";
                    else if (m_pieces[3][1].test(index))
                        out += "r";
                    else if (m_pieces[4][1].test(index))
                        out += "q";
                    else if (m_pieces[5][1].test(index))
                        out += "k";
                }
            }
        }
        if (n_empty != 0)
            out += std::to_string(n_empty);
        if (i > 0)
            out += "/";
    }

    // Turn
    if (m_turn == Turn::WHITE)
        out += " w ";
    else
        out += " b ";

    // Castling rights
    if (m_castling_rights[0][0])
        out += "K";
    if (m_castling_rights[1][0])
        out += "Q";
    if (m_castling_rights[0][1])
        out += "k";
    if (m_castling_rights[1][1])
        out += "q";
    if (!(m_castling_rights[0][0] || m_castling_rights[1][0]
       || m_castling_rights[0][1] || m_castling_rights[1][1]))
        out += "-";

    // En passant
    out += " ";
    if (m_enpassant_square == -1)
        out += "-";
    else
        out += get_square(m_enpassant_square);

    // Half-move clock
    out += " " + std::to_string(m_half_move_clock);

    // Full-move clock
    out += " " + std::to_string(m_full_move_clock);

    return out;
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


void Board::init()
{
    // Pieces
    for (Square square = 0; square < NUM_SQUARES; square++)
        m_board_pieces[square] = NO_PIECE;
    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (Turn turn : { WHITE, BLACK })
        {
            Bitboard bb = get_pieces(turn, piece);
            while (bb)
                m_board_pieces[bb.bitscan_forward_reset()] = get_piece(piece, turn);
        }

    // Generate hash
    m_hash = generate_hash();

    // Checkers
    update_checkers();

    // Material and phase evaluation
    m_phase = Phases::Total;
    for (PieceType piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (Turn turn : { WHITE, BLACK })
        {
            Bitboard bb = get_pieces(turn, piece);
            m_eval1 += piece_value[piece] * bb.count() * turn_to_color(turn);
            m_phase -= bb.count() * Phases::Pieces[piece];
            while (bb)
                m_eval1 += piece_square(piece, bb.bitscan_forward_reset(), turn) * turn_to_color(turn);
        }
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
    {
        std::cout << "bad check on moved side" << std::endl;
        return false;
    }

    // Bitboard consistency
    Bitboard occupancy;
    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
        for (auto turn : { WHITE, BLACK })
        {
            if (m_pieces[piece][turn] & occupancy)
            {
                std::cout << "bad occupancy" << std::endl;
                std::cout << (int)piece << " " << (int)turn << std::endl;
                std::cout << m_pieces[piece][turn] << std::endl;
                std::cout << occupancy << std::endl;
                return false;
            }
            occupancy |= m_pieces[piece][turn];
        }

    // Piece-square consistency
    for (Square square = 0; square < NUM_SQUARES; square++)
        if (m_board_pieces[square] == NO_PIECE)
        {
            if (occupancy.test(square))
            {
                std::cout << "bad empty square" << std::endl;
                std::cout << (int)square << std::endl;
                std::cout << (int)m_board_pieces[square] << std::endl;
                std::cout << occupancy << std::endl;
                return false;
            }
        }
        else
        {
            auto piece = get_piece_at(square);
            auto turn = (m_board_pieces[square] % 2 == 0) ? WHITE : BLACK;
            if (!m_pieces[piece][turn].test(square))
            {
                std::cout << "bad occupied square" << std::endl;
                std::cout << (int)square << std::endl;
                std::cout << (int)piece << std::endl;
                std::cout << (int)turn << std::endl;
                std::cout << (int)m_board_pieces[square] << std::endl;
                std::cout << m_pieces[piece][turn] << std::endl;
                return false;
            }
        }

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
    // Static-Exchange evaluation with prunning
    constexpr Score piece_score[] = { 10, 30, 30, 50, 90, 1000, 0, 0 };

    Square target = move.to();

    // Make the initial capture
    PieceType last_attacker = get_piece_at(move.from());
    Score gain = piece_score[move.is_ep_capture() ? PAWN : get_piece_at(target)] - threshold / 10;
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
            return 10 * gain;

        // Get least valuable attacker
        Square attacker = least_valuable(attacks_target);
        Bitboard attacker_bb = Bitboard::from_square(attacker);

        // Make the capture
        gain += color * piece_score[last_attacker];
        last_attacker = get_piece_at(attacker);
        occupancy ^= attacker_bb;
        side_to_move = ~side_to_move;
        color = -color;

        // Get opponent attackers
        attacks_target = attackers(target, occupancy, side_to_move) & occupancy;
    }

    return 10 * gain;
}


MixedScore Board::material_eval() const
{
    return m_eval1;
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


bool Position::is_check() const
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


std::ostream& operator<<(std::ostream& out, const Board& board)
{
    Bitboard white_pieces = board.m_pieces[0][0] | board.m_pieces[1][0] | board.m_pieces[2][0]
                          | board.m_pieces[3][0] | board.m_pieces[4][0] | board.m_pieces[5][0];
    Bitboard black_pieces = board.m_pieces[0][1] | board.m_pieces[1][1] | board.m_pieces[2][1]
                          | board.m_pieces[3][1] | board.m_pieces[4][1] | board.m_pieces[5][1];
    Bitboard occupied = white_pieces | black_pieces;

    for (int i = 7; i >= 0; i--)
    {
        for (int j = 0; j < 8; j++)
        {
            int index = i * 8 + j;
            if (!occupied.test(index))
            {
                out << " .";
            }
            else
            {
                if (white_pieces.test(index))
                {
                    if (board.m_pieces[0][0].test(index))
                        out << " P";
                    else if (board.m_pieces[1][0].test(index))
                        out << " N";
                    else if (board.m_pieces[2][0].test(index))
                        out << " B";
                    else if (board.m_pieces[3][0].test(index))
                        out << " R";
                    else if (board.m_pieces[4][0].test(index))
                        out << " Q";
                    else if (board.m_pieces[5][0].test(index))
                        out << " K";
                }
                else
                {
                    if (board.m_pieces[0][1].test(index))
                        out << " p";
                    else if (board.m_pieces[1][1].test(index))
                        out << " n";
                    else if (board.m_pieces[2][1].test(index))
                        out << " b";
                    else if (board.m_pieces[3][1].test(index))
                        out << " r";
                    else if (board.m_pieces[4][1].test(index))
                        out << " q";
                    else if (board.m_pieces[5][1].test(index))
                        out << " k";
                }
            }
        }
        out << "\n";
    }
    return out;
}
