#include "Types.hpp"
#include "Bitboard.hpp"
#include "Position.hpp"
#include "Evaluation.hpp"
#include "PieceSquareTables.hpp"
#include <cassert>
#include <stdlib.h>


MixedScore material(Board board)
{
    MixedScore result(0, 0);

    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
        result += piece_value[piece] * (board.get_pieces(WHITE, piece).count() -
            board.get_pieces(BLACK, piece).count());

    return result;
}


MixedScore piece_square_value(Board board, int eg)
{
    Bitboard bb;
    S sum(0, 0);

    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
    {
        for (auto turn : { WHITE, BLACK })
        {
            Color color = turn_to_color(turn);
            bb = board.get_pieces(turn, piece);
            while (bb)
                sum += piece_square(piece, bb.bitscan_forward_reset(), turn) * color;
        }
    }

    return sum;
}


MixedScore pawn_structure(Bitboard white_pawns, Bitboard black_pawns)
{
    constexpr MixedScore DoubledPenalty(13, 51);
    constexpr MixedScore IsolatedPenalty(3, 15);
    constexpr MixedScore BackwardPenalty(9, 22);
    constexpr MixedScore IslandPenalty(3, 12);
    constexpr MixedScore PassedBonus[] = { MixedScore(  0,   0), MixedScore(  0,   0),
                                           MixedScore(  7,  27), MixedScore( 16,  32),
                                           MixedScore( 17,  40), MixedScore( 64,  71),
                                           MixedScore(170, 174), MixedScore(278, 262)  };

    constexpr Direction Up = 8;
    constexpr Direction Left = -1;
    constexpr Direction Right = 1;

    Bitboard white_front_span = white_pawns.fill< Up>();
    Bitboard black_front_span = black_pawns.fill<-Up>();
    Bitboard white_back_span  = white_pawns.fill<-Up>();
    Bitboard black_back_span  = black_pawns.fill< Up>();

    Bitboard white_file_span = white_front_span | white_back_span;
    Bitboard black_file_span = black_front_span | black_back_span;

    Bitboard white_attack_span = (white_front_span & ~Bitboards::a_file).shift<Left >() |
                                 (white_front_span & ~Bitboards::h_file).shift<Right>();
    Bitboard black_attack_span = (black_front_span & ~Bitboards::a_file).shift<Left >() |
                                 (black_front_span & ~Bitboards::h_file).shift<Right>();

    Bitboard white_attack_span_file = (white_file_span & ~Bitboards::a_file).shift<Left >() |
                                      (white_file_span & ~Bitboards::h_file).shift<Right>();
    Bitboard black_attack_span_file = (black_file_span & ~Bitboards::a_file).shift<Left >() |
                                      (black_file_span & ~Bitboards::h_file).shift<Right>();

    Bitboard white_doubled_pawns = white_pawns & white_pawns.fill_excluded<-Up>();
    Bitboard black_doubled_pawns = black_pawns & black_pawns.fill_excluded< Up>();

    Bitboard white_passed_pawns = white_pawns & ~(black_front_span | black_attack_span.shift<-Up>()) & ~white_doubled_pawns;
    Bitboard black_passed_pawns = black_pawns & ~(white_front_span | white_attack_span.shift< Up>()) & ~black_doubled_pawns;

    Bitboard white_isolated_pawns = white_pawns & ~white_attack_span_file;
    Bitboard black_isolated_pawns = black_pawns & ~black_attack_span_file;

    Bitboard white_backward_pawns = white_pawns & ~white_isolated_pawns & ~white_attack_span;
    Bitboard black_backward_pawns = black_pawns & ~black_isolated_pawns & ~black_attack_span;

    int white_islands = (~white_file_span & Bitboards::rank_1).count();
    int black_islands = (~black_file_span & Bitboards::rank_1).count();

    // Build passed pawn scores
    MixedScore result(0, 0);
    while (white_passed_pawns)
        result += PassedBonus[rank(white_passed_pawns.bitscan_forward_reset())];
    while (black_passed_pawns)
        result -= PassedBonus[7 - rank(black_passed_pawns.bitscan_forward_reset())];

    // Penalty for non-pushed central pawns
    result += MixedScore(-35, -50) * (white_pawns.test(SQUARE_E2) + white_pawns.test(SQUARE_D2));
    result -= MixedScore(-35, -50) * (black_pawns.test(SQUARE_E7) + black_pawns.test(SQUARE_D7));

    return result
         - IsolatedPenalty * (white_isolated_pawns.count() - black_isolated_pawns.count())
         - BackwardPenalty * (white_backward_pawns.count() - black_backward_pawns.count())
         - DoubledPenalty * (white_doubled_pawns.count() - black_doubled_pawns.count())
         - IslandPenalty * (white_islands - black_islands);
}


template <PieceType PIECE_TYPE, Turn TURN>
MixedScore piece(const Board& board, Bitboard occupancy, Bitboard filter, Bitboard safe_squares_from_pawns, Bitboard our_pawn_attacks)
{
    constexpr Direction Up = (TURN == WHITE) ? 8 : -8;
    constexpr Bitboard rank7 = (TURN == WHITE) ? Bitboards::rank_7 : Bitboards::rank_2;

    constexpr MixedScore score_per_move(10, 10);
    constexpr MixedScore nominal_moves = PIECE_TYPE == KNIGHT ? MixedScore(4, 4)
                                       : PIECE_TYPE == BISHOP ? MixedScore(5, 5)
                                       : PIECE_TYPE == ROOK   ? MixedScore(4, 6)
                                       :                        MixedScore(7, 9); // QUEEN
    MixedScore mobility(0, 0);
    MixedScore placement(0, 0);

    Bitboard b = board.get_pieces<TURN, PIECE_TYPE>();
    while (b)
    {
        Square square = b.bitscan_forward_reset();
        Bitboard attacks = Bitboards::get_attacks<PIECE_TYPE>(square, occupancy) & filter;

        int safe_squares = attacks.count();

        mobility += (MixedScore(safe_squares, safe_squares) - nominal_moves) * score_per_move;

        // TODO: other terms
        if (PIECE_TYPE == KNIGHT)
        {

        }
        else if (PIECE_TYPE == BISHOP)
        {

        }
        else if (PIECE_TYPE == ROOK)
        {
            // Connects to another rook?
            placement += MixedScore(15, 5) * (b & attacks).count();
        }
        else if (PIECE_TYPE == QUEEN)
        {

        }
    }

    // Set-wise terms
    MixedScore group_scores(0, 0);
    b = board.get_pieces<TURN, PIECE_TYPE>();
    if (PIECE_TYPE == KNIGHT)
    {
        // Behind enemy lines?
        placement += MixedScore(25, 20) * (b & safe_squares_from_pawns).count();
        // Defended by pawns?
        placement += MixedScore(5, 1) * (b & our_pawn_attacks).count();
        // Additional bonus if both previous conditions apply
        placement += MixedScore(25, 10) * (b & safe_squares_from_pawns & our_pawn_attacks).count();
    }
    else if (PIECE_TYPE == BISHOP)
    {
        // Behind enemy lines?
        placement += MixedScore(25, 20) * (b & safe_squares_from_pawns).count();
        // Defended by pawns?
        placement += MixedScore(5, 1) * (b & our_pawn_attacks).count();
        // Additional bonus if both previous conditions apply
        placement += MixedScore(25, 10) * (b & safe_squares_from_pawns & our_pawn_attacks).count();

        // Bishop pair?
        if ((b & Bitboards::square_color[WHITE]) && (b & Bitboards::square_color[BLACK]))
            group_scores += MixedScore(10, 20);
    }
    else if (PIECE_TYPE == ROOK)
    {
        // Rooks on 7th rank?
        placement += MixedScore(10, 5) * (b & rank7).count();
        // Files for each rook: check if open or semi-open
        Bitboard files = b.fill<Up>();
        placement += MixedScore(20, 7) * (2 * b.count() - (files & (board.get_pieces<WHITE, PAWN>() | board.get_pieces<BLACK, PAWN>())).count());
    }
    else if (PIECE_TYPE == QUEEN)
    {

    }

    return mobility + placement;
}


MixedScore pieces(const Board& board)
{
    constexpr Direction Up = 8;
    constexpr Direction Left = -1;
    constexpr Direction Right = 1;

    Bitboard white_pawn_attacks = (board.get_pieces<WHITE, PAWN>() & ~Bitboards::a_file).shift< Up + Left >()
                                | (board.get_pieces<WHITE, PAWN>() & ~Bitboards::h_file).shift< Up + Right>();
    Bitboard black_pawn_attacks = (board.get_pieces<BLACK, PAWN>() & ~Bitboards::a_file).shift<-Up + Left >()
                                | (board.get_pieces<BLACK, PAWN>() & ~Bitboards::h_file).shift<-Up + Right>();

    Bitboard white_pawn_attack_span = white_pawn_attacks.fill< Up>();
    Bitboard black_pawn_attack_span = black_pawn_attacks.fill<-Up>();

    Bitboard white_filter = ~black_pawn_attacks | ~board.get_pieces<WHITE>();
    Bitboard black_filter = ~white_pawn_attacks | ~board.get_pieces<BLACK>();
    Bitboard occupancy = board.get_pieces<WHITE>() | board.get_pieces<BLACK>();

    MixedScore result = piece<KNIGHT, WHITE>(board, occupancy, white_filter, ~black_pawn_attack_span, white_pawn_attacks)
                      + piece<BISHOP, WHITE>(board, occupancy, white_filter, ~black_pawn_attack_span, white_pawn_attacks)
                      + piece<  ROOK, WHITE>(board, occupancy, white_filter, ~black_pawn_attack_span, white_pawn_attacks)
                      + piece< QUEEN, WHITE>(board, occupancy, white_filter, ~black_pawn_attack_span, white_pawn_attacks)
                      - piece<KNIGHT, BLACK>(board, occupancy, black_filter, ~white_pawn_attack_span, black_pawn_attacks)
                      - piece<BISHOP, BLACK>(board, occupancy, black_filter, ~white_pawn_attack_span, black_pawn_attacks)
                      - piece<  ROOK, BLACK>(board, occupancy, black_filter, ~white_pawn_attack_span, black_pawn_attacks)
                      - piece< QUEEN, BLACK>(board, occupancy, black_filter, ~white_pawn_attack_span, black_pawn_attacks);

    return result;
}


MixedScore king_safety(const Board& board)
{
    constexpr Direction Up = 8;
    constexpr Direction Left = -1;
    constexpr Direction Right = 1;

    constexpr MixedScore pawn_shelter[10] = { MixedScore(-100,   0), MixedScore(-25,   0), MixedScore( 0,   0),
                                              MixedScore(  25,   0), MixedScore( 35,  -5), MixedScore(40,  -5),
                                              MixedScore(  40, -10), MixedScore( 41, -15), MixedScore(42, -20), 
                                              MixedScore(  43, -25) };

    constexpr MixedScore king_attacks[5] = { MixedScore(0, 0), MixedScore(10, 5), MixedScore(50, 0), MixedScore(100, 0), MixedScore(200, 0) };
    constexpr MixedScore king_slider_attacks[4] = { MixedScore(-150, -100), MixedScore(-50, -20), MixedScore(-15, -2), MixedScore(0, 0) };

    Bitboard occupancy = board.get_pieces();

    Bitboard white_king = board.get_pieces<WHITE, KING>();
    Bitboard black_king = board.get_pieces<BLACK, KING>();

    Bitboard white_mask = Bitboards::get_attacks<KING>(white_king.bitscan_forward(), occupancy) | white_king;
    Bitboard black_mask = Bitboards::get_attacks<KING>(black_king.bitscan_forward(), occupancy) | black_king;

    Bitboard white_pawns = board.get_pieces<WHITE, PAWN>();
    Bitboard black_pawns = board.get_pieces<BLACK, PAWN>();

    MixedScore king_shelter = pawn_shelter[((white_mask | white_mask.shift< 2 * Up>()) & white_pawns).count()]
                            - pawn_shelter[((white_mask | black_mask.shift<-2 * Up>()) & black_pawns).count()];

    // Back-rank bonus
    king_shelter += MixedScore(50, -50) * (white_king & Bitboards::rank_1).count();
    king_shelter -= MixedScore(50, -50) * (black_king & Bitboards::rank_8).count();

    // X-rays with enemy sliders?
    Bitboard white_rooks   = board.get_pieces<WHITE, ROOK>()   | board.get_pieces<WHITE, QUEEN>();
    Bitboard white_bishops = board.get_pieces<WHITE, BISHOP>() | board.get_pieces<WHITE, QUEEN>();
    Bitboard black_rooks   = board.get_pieces<BLACK, ROOK>()   | board.get_pieces<BLACK, QUEEN>();
    Bitboard black_bishops = board.get_pieces<BLACK, BISHOP>() | board.get_pieces<BLACK, QUEEN>();

    MixedScore x_rays(0, 0);
    Square white_king_sq = white_king.bitscan_forward();
    Square black_king_sq = black_king.bitscan_forward();
    Bitboard white_pieces = board.get_pieces<WHITE>();
    Bitboard black_pieces = board.get_pieces<BLACK>();
    Bitboard white_slider_attackers = (Bitboards::ranks_files[white_king.bitscan_forward()] & black_rooks)
                                    | (Bitboards::diagonals  [white_king.bitscan_forward()] & black_bishops);
    Bitboard black_slider_attackers = (Bitboards::ranks_files[black_king.bitscan_forward()] & white_rooks)
                                    | (Bitboards::diagonals  [black_king.bitscan_forward()] & white_bishops);
    while (white_slider_attackers)
        x_rays += king_slider_attacks[std::min(3, Bitboards::between(white_king_sq, white_slider_attackers.bitscan_forward_reset()).count())];
    while (black_slider_attackers)
        x_rays -= king_slider_attacks[std::min(3, Bitboards::between(black_king_sq, black_slider_attackers.bitscan_forward_reset()).count())];


    // Attackers to the squares near the king
    int white_attacked = 0;
    int black_attacked = 0;
    while (white_mask)
        white_attacked += board.attackers<BLACK>(white_mask.bitscan_forward_reset(), occupancy).count();
    while (black_mask)
        black_attacked += board.attackers<WHITE>(black_mask.bitscan_forward_reset(), occupancy).count();

    MixedScore king_threats = king_attacks[std::min(4, white_attacked / 2)] - king_attacks[std::min(4, black_attacked / 2)];


    // King out in the open?
    Bitboard white_slides = Bitboards::get_attacks<BISHOP>(white_king.bitscan_forward(), occupancy)
                          | Bitboards::get_attacks<  ROOK>(white_king.bitscan_forward(), occupancy);
    Bitboard black_slides = Bitboards::get_attacks<BISHOP>(black_king.bitscan_forward(), occupancy)
                          | Bitboards::get_attacks<  ROOK>(black_king.bitscan_forward(), occupancy);
    int white_possible_dirs = white_mask.count() - 1;
    int black_possible_dirs = black_mask.count() - 1;
    int white_safe_dirs = (white_slides & board.get_pieces<WHITE>()).count();
    int black_safe_dirs = (black_slides & board.get_pieces<BLACK>()).count();
    king_threats += MixedScore(-15, 8) * (white_possible_dirs - white_safe_dirs - black_possible_dirs + black_safe_dirs);



    // Open or semi-open files near the king
    Bitboard white_file_king  = white_king.fill< Up>();
    Bitboard black_file_king  = black_king.fill<-Up>();
    Bitboard white_file_left  = (white_king & ~Bitboards::a_file).shift< Left>().fill< Up>();
    Bitboard black_file_left  = (black_king & ~Bitboards::a_file).shift< Left>().fill<-Up>();
    Bitboard white_file_right = (white_king & ~Bitboards::h_file).shift<Right>().fill< Up>();
    Bitboard black_file_right = (black_king & ~Bitboards::h_file).shift<Right>().fill<-Up>();
    king_threats += MixedScore(-75, 0) * !(white_file_king  & board.get_pieces<WHITE, PAWN>());
    king_threats -= MixedScore(-75, 0) * !(black_file_king  & board.get_pieces<BLACK, PAWN>());
    king_threats += MixedScore(-35, 0) * !(white_file_left  & board.get_pieces<WHITE, PAWN>());
    king_threats -= MixedScore(-35, 0) * !(black_file_left  & board.get_pieces<BLACK, PAWN>());
    king_threats += MixedScore(-35, 0) * !(white_file_right & board.get_pieces<WHITE, PAWN>());
    king_threats -= MixedScore(-35, 0) * !(white_file_right & board.get_pieces<BLACK, PAWN>());


    return king_shelter + x_rays + king_threats;
}


Score evaluation(const Position& position, bool output)
{
    auto board = position.board();
    MixedScore mixed_result(0, 0);

    // Material and PSQT: incrementally updated in the position
    MixedScore tmp = board.material_eval() / 10;
    mixed_result += tmp;
    if (output)
        std::cout << "Material    " << (int)tmp.middlegame() << " " << (int)tmp.endgame() << std::endl;

    // Pawn structure
    tmp = pawn_structure(board.get_pieces<WHITE, PAWN>(), board.get_pieces < BLACK, PAWN>());
    mixed_result += tmp;
    if (output)
        std::cout << "Pawns       " << (int)tmp.middlegame() << " " << (int)tmp.endgame() << std::endl;

    // Mobility
    tmp = pieces(board);
    mixed_result += tmp;
    if (output)
        std::cout << "Mobility    " << (int)tmp.middlegame() << " " << (int)tmp.endgame() << std::endl;

    // King safety
    tmp = king_safety(board);
    mixed_result += tmp;
    if (output)
        std::cout << "King safety " << (int)tmp.middlegame() << " " << (int)tmp.endgame() << std::endl;

    // Tapered eval
    Score result = mixed_result.tapered(board.phase());
    if (output)
        std::cout << "Phase       " << (int)board.phase() << std::endl;
    if (output)
        std::cout << "Score       " << (int)result << std::endl;

    // We don't return exact draw scores -> add one centipawn to the moving side
    if (result == SCORE_DRAW)
        result += turn_to_color(board.turn());

    return result;
}
