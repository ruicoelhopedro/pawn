#include "Types.hpp"
#include "Bitboard.hpp"
#include "Position.hpp"
#include "Evaluation.hpp"
#include "Endgame.hpp"
#include "PieceSquareTables.hpp"
#include <cassert>
#include <stdlib.h>
#include <iomanip>


Attacks::Attacks()
{
    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
    {
        m_attacks[piece] = Bitboard();
        m_double_attacks[piece] = Bitboard();
    }
}


MixedScore material(Board board, EvalData& eval)
{
    eval.fields[WHITE].material = MixedScore(0, 0);
    eval.fields[BLACK].material = MixedScore(0, 0);

    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
    {
        eval.fields[WHITE].material += piece_value[piece] * board.get_pieces(WHITE, piece).count();
        eval.fields[BLACK].material += piece_value[piece] * board.get_pieces(BLACK, piece).count();
    }
    return eval.fields[WHITE].material - eval.fields[BLACK].material;
}


MixedScore piece_square_value(Board board, EvalData& eval)
{
    eval.fields[WHITE].placement = MixedScore(0, 0);
    eval.fields[BLACK].placement = MixedScore(0, 0);
    Bitboard bb;

    for (auto piece : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
    {
        for (auto turn : { WHITE, BLACK })
        {
            bb = board.get_pieces(turn, piece);
            while (bb)
                eval.fields[turn].placement += piece_square(piece, bb.bitscan_forward_reset(), turn);
        }
    }

    return eval.fields[WHITE].placement - eval.fields[BLACK].placement;
}


MixedScore pawn_structure(const Board& board, EvalData& eval)
{
    constexpr MixedScore DoubledPenalty(13, 51);
    constexpr MixedScore IsolatedPenalty(3, 15);
    constexpr MixedScore BackwardPenalty(9, 22);
    constexpr MixedScore IslandPenalty(3, 12);
    constexpr MixedScore NonPushedCentralPenalty(35, 50);
    constexpr MixedScore PassedBonus[] = { MixedScore(  0,   0), MixedScore(  0,   0),
                                           MixedScore(  7,  27), MixedScore( 16,  32),
                                           MixedScore( 17,  40), MixedScore( 64,  71),
                                           MixedScore(170, 174), MixedScore(278, 262) };

    constexpr Direction Up = 8;
    // constexpr Direction Left = -1;
    // constexpr Direction Right = 1;

    const Bitboard pawns[] = { board.get_pieces<WHITE, PAWN>(),
                               board.get_pieces<BLACK, PAWN>() };
    //const Bitboard occupancy = pawns[WHITE] | pawns[BLACK];

    // Attacks
    eval.pawns[WHITE].attacks = Bitboards::get_attacks_pawns<WHITE>(pawns[WHITE]);
    eval.pawns[BLACK].attacks = Bitboards::get_attacks_pawns<BLACK>(pawns[BLACK]);
    eval.attacks[WHITE].push<PAWN>(eval.pawns[WHITE].attacks);
    eval.attacks[BLACK].push<PAWN>(eval.pawns[BLACK].attacks);

    // Attackable squares (attack span)
    eval.pawns[WHITE].attackable = eval.pawns[WHITE].attacks.fill< Up>();
    eval.pawns[BLACK].attackable = eval.pawns[BLACK].attacks.fill<-Up>();

    // Open files
    eval.pawns[WHITE].open_files = ~(pawns[WHITE].fill_file());
    eval.pawns[BLACK].open_files = ~(pawns[BLACK].fill_file());

    // Isolated pawns
    eval.pawns[WHITE].isolated = pawns[WHITE] & ~(eval.pawns[WHITE].attackable.fill_file());
    eval.pawns[BLACK].isolated = pawns[BLACK] & ~(eval.pawns[BLACK].attackable.fill_file());

    // Blocked pawns
    eval.pawns[WHITE].blocked = pawns[WHITE] & pawns[BLACK].shift<-Up>();
    eval.pawns[BLACK].blocked = pawns[BLACK] & pawns[WHITE].shift< Up>();

    // Doubled pawns (this does not pick the most advanced doubled pawn, only flags the ones behind it)
    eval.pawns[WHITE].doubled = pawns[WHITE] & pawns[WHITE].fill_excluded<-Up>();
    eval.pawns[BLACK].doubled = pawns[BLACK] & pawns[BLACK].fill_excluded< Up>();

    // Passed pawns (not doubled and not attacked by or in front of enemy pawns)
    eval.pawns[WHITE].passed = pawns[WHITE] & ~eval.pawns[WHITE].doubled & ~(eval.pawns[BLACK].attackable | eval.pawns[BLACK].open_files);
    eval.pawns[BLACK].passed = pawns[BLACK] & ~eval.pawns[BLACK].doubled & ~(eval.pawns[WHITE].attackable | eval.pawns[WHITE].open_files);

    // Backward pawns
    eval.pawns[WHITE].backward = pawns[WHITE] & ~(eval.pawns[WHITE].attackable.shift<-Up>()) & eval.pawns[BLACK].attacks.shift<-Up>();
    eval.pawns[BLACK].backward = pawns[BLACK] & ~(eval.pawns[BLACK].attackable.shift< Up>()) & eval.pawns[WHITE].attacks.shift< Up>();

    // Islands
    eval.pawns[WHITE].islands = (eval.pawns[WHITE].open_files & Bitboards::rank_1).count();
    eval.pawns[BLACK].islands = (eval.pawns[BLACK].open_files & Bitboards::rank_1).count();

    // Outposts
    eval.pawns[WHITE].outposts = pawns[WHITE].shift< Up>() & ~eval.pawns[WHITE].attackable;
    eval.pawns[BLACK].outposts = pawns[BLACK].shift<-Up>() & ~eval.pawns[BLACK].attackable;

    // Levers
    eval.pawns[WHITE].levers = pawns[WHITE] & eval.pawns[BLACK].attacks;
    eval.pawns[BLACK].levers = pawns[BLACK] & eval.pawns[WHITE].attacks;

    // Passer candidates
    eval.pawns[WHITE].candidate_passer = pawns[WHITE] & eval.pawns[BLACK].open_files;
    eval.pawns[BLACK].candidate_passer = pawns[BLACK] & eval.pawns[WHITE].open_files;

    // Squares behind our pawns
    eval.pawns[WHITE].behind = ~(eval.pawns[WHITE].attackable | pawns[WHITE].fill_excluded< Up>());
    eval.pawns[BLACK].behind = ~(eval.pawns[BLACK].attackable | pawns[BLACK].fill_excluded<-Up>());

    // Init output fields with penalty for non-pushed central pawns
    eval.fields[WHITE].pawns = NonPushedCentralPenalty * -(pawns[WHITE].test(SQUARE_E2) + pawns[WHITE].test(SQUARE_D2));
    eval.fields[BLACK].pawns = NonPushedCentralPenalty * -(pawns[BLACK].test(SQUARE_E7) + pawns[BLACK].test(SQUARE_D7));

    // Build passed pawn scores
    Bitboard bw = eval.pawns[WHITE].passed;
    Bitboard bb = eval.pawns[BLACK].passed;
    while (bw)
        eval.fields[WHITE].pawns += PassedBonus[rank_relative<WHITE>(bw.bitscan_forward_reset())];
    while (bb)
        eval.fields[BLACK].pawns += PassedBonus[rank_relative<BLACK>(bb.bitscan_forward_reset())];

    // Various penalties for both sides
    for (auto turn : { WHITE, BLACK })
        eval.fields[turn].pawns -= IsolatedPenalty * eval.pawns[turn].isolated.count()
                                 + BackwardPenalty * eval.pawns[turn].backward.count()
                                 +  DoubledPenalty * eval.pawns[turn].doubled.count()
                                 +   IslandPenalty * eval.pawns[turn].islands;

    // Final score
    return eval.fields[WHITE].pawns - eval.fields[BLACK].pawns;
}


template <PieceType PIECE_TYPE, Turn TURN>
MixedScore piece(const Board& board, Bitboard occupancy, Bitboard filter, EvalData& eval)
{
    constexpr Bitboard rank1 = (TURN == WHITE) ? Bitboards::rank_1 : Bitboards::rank_8;
    constexpr Bitboard rank7 = (TURN == WHITE) ? Bitboards::rank_7 : Bitboards::rank_2;

    constexpr MixedScore score_per_move(5, 8);
    constexpr MixedScore nominal_moves = PIECE_TYPE == KNIGHT ? MixedScore(4, 4)
                                       : PIECE_TYPE == BISHOP ? MixedScore(5, 5)
                                       : PIECE_TYPE == ROOK   ? MixedScore(4, 6)
                                       :                        MixedScore(7, 9); // QUEEN

    constexpr MixedScore Outpost(50, 40);
    constexpr MixedScore ReachableOutpost(20, 10);
    MixedScore mobility(0, 0);
    MixedScore placement(0, 0);

    Bitboard b = board.get_pieces<TURN, PIECE_TYPE>();
    while (b)
    {
        Square square = b.bitscan_forward_reset();
        Bitboard attacks = Bitboards::get_attacks<PIECE_TYPE>(square, occupancy);

        // Attacks to the king ring
        if (attacks & eval.king_zone[~TURN])
            eval.king_attackers[TURN].set(square);

        eval.attacks[TURN].push<PIECE_TYPE>(attacks);
        attacks &= filter;
        int safe_squares = attacks.count();

        mobility += (MixedScore(safe_squares, safe_squares) - nominal_moves) * score_per_move;

        // TODO: other terms
        if (PIECE_TYPE == KNIGHT)
        {
            // Can reach an outpost?
            if (attacks & eval.pawns[~TURN].outposts)
                placement += ReachableOutpost;
        }
        else if (PIECE_TYPE == BISHOP)
        {
            // Can reach an outpost?
            if (attacks & eval.pawns[~TURN].outposts)
                placement += ReachableOutpost;
        }
        else if (PIECE_TYPE == ROOK)
        {
            // Connects to another rook?
            placement += MixedScore(15, 5) * (b & attacks).count();

            // Can safely reach an open file?
            if (attacks & eval.pawns[TURN].open_files)
                placement += MixedScore(15, 5);

            // // King between rooks on first rank?
            // if (b && rank1.test(square))
            // {
            //     Square other_rook = b.bitscan_forward();
            //     Square king = board.get_pieces<TURN, KING>();
            //     if (rank1.test(other_rook) && Bitboards::between(square, other_rook).test(king))
            //         placement += MixedScore(-25, -50);
            // }
        }
        else if (PIECE_TYPE == QUEEN)
        {

        }
    }

    // General placement terms
    b = board.get_pieces<TURN, PIECE_TYPE>();
    // Behind enemy lines?
    placement += MixedScore(25, 4) * (b & eval.pawns[~TURN].behind).count();

    // Set-wise terms
    if (PIECE_TYPE == KNIGHT)
    {
        // In an outpost?
        placement += Outpost * (b & eval.pawns[~TURN].outposts).count();
    }
    else if (PIECE_TYPE == BISHOP)
    {
        // In an outpost?
        placement += Outpost * (b & eval.pawns[~TURN].outposts).count();

        // Bishop pair?
        if ((b & Bitboards::square_color[WHITE]) && (b & Bitboards::square_color[BLACK]))
            placement += MixedScore(10, 20);
    }
    else if (PIECE_TYPE == ROOK)
    {
        // Rooks on 7th rank?
        placement += MixedScore(10, 5) * (b & rank7).count();

        // Files for each rook: check if open or semi-open
        Bitboard semi_open = eval.pawns[TURN].open_files;
        Bitboard open = semi_open & eval.pawns[~TURN].open_files;
        placement += MixedScore(20, 7) * (b & semi_open).count();
        placement += MixedScore(20, 7) * (b & open).count();
    }
    else if (PIECE_TYPE == QUEEN)
    {

    }

    return mobility + placement;
}


MixedScore pieces(const Board& board, EvalData& eval)
{
    // constexpr Direction Up = 8;
    // constexpr Direction Left = -1;
    // constexpr Direction Right = 1;

    Bitboard white_pawn_attacks = eval.attacks[WHITE].get<PAWN>();
    Bitboard black_pawn_attacks = eval.attacks[BLACK].get<PAWN>();

    Bitboard white_filter = ~black_pawn_attacks | ~board.get_pieces<WHITE>();
    Bitboard black_filter = ~white_pawn_attacks | ~board.get_pieces<BLACK>();
    Bitboard occupancy = board.get_pieces<WHITE>() | board.get_pieces<BLACK>();

    // Knights and Bishops
    eval.fields[WHITE].mobility += piece<KNIGHT, WHITE>(board, occupancy, white_filter, eval);
    eval.fields[BLACK].mobility += piece<KNIGHT, BLACK>(board, occupancy, black_filter, eval);
    eval.fields[WHITE].mobility += piece<BISHOP, WHITE>(board, occupancy, white_filter, eval);
    eval.fields[BLACK].mobility += piece<BISHOP, BLACK>(board, occupancy, black_filter, eval);
    white_filter &= ~eval.attacks[BLACK].get();
    black_filter &= ~eval.attacks[WHITE].get();

    // Rooks
    eval.fields[WHITE].mobility += piece<ROOK, WHITE>(board, occupancy, white_filter, eval);
    eval.fields[BLACK].mobility += piece<ROOK, BLACK>(board, occupancy, black_filter, eval);
    white_filter &= ~eval.attacks[BLACK].get();
    black_filter &= ~eval.attacks[WHITE].get();

    // Queens
    eval.fields[WHITE].mobility += piece<QUEEN, WHITE>(board, occupancy, white_filter, eval);
    eval.fields[BLACK].mobility += piece<QUEEN, BLACK>(board, occupancy, black_filter, eval);
    white_filter &= ~eval.attacks[BLACK].get();
    black_filter &= ~eval.attacks[WHITE].get();

    Bitboard white_control = eval.attacks[WHITE].get() & ~eval.attacks[BLACK].get();
    Bitboard black_control = eval.attacks[BLACK].get() & ~eval.attacks[WHITE].get();

    // King mobility
    Square white_king = board.get_pieces<WHITE, KING>().bitscan_forward();
    Square black_king = board.get_pieces<BLACK, KING>().bitscan_forward();
    Bitboard white_king_safe_squares = Bitboards::get_attacks<KING>(white_king, occupancy) & white_filter;
    Bitboard black_king_safe_squares = Bitboards::get_attacks<KING>(black_king, occupancy) & black_filter;
    eval.attacks[WHITE].push<KING>(white_king_safe_squares);
    eval.attacks[BLACK].push<KING>(black_king_safe_squares);

    // Global mobility
    eval.fields[WHITE].mobility += MixedScore(0, 100) * (std::min(10, eval.attacks[WHITE].get().count()) - 10);
    eval.fields[BLACK].mobility += MixedScore(0, 100) * (std::min(10, eval.attacks[BLACK].get().count()) - 10);

    // Space and square control
    Bitboard center = Bitboards::zone1 | Bitboards::zone2;
    eval.fields[WHITE].mobility += MixedScore(15, 1) * (white_control & center).count();
    eval.fields[BLACK].mobility += MixedScore(15, 1) * (black_control & center).count();

    return eval.fields[WHITE].mobility - eval.fields[BLACK].mobility;
}


MixedScore king_safety(const Board& board, EvalData& eval)
{
    constexpr Direction Up = 8;
    //constexpr Direction Left = -1;
    //constexpr Direction Right = 1;

    constexpr MixedScore pawn_shelter[10] = { MixedScore(-100,   0), MixedScore(-25,   0), MixedScore( 0,   0),
                                              MixedScore(  25,   0), MixedScore( 35,  -5), MixedScore(40,  -5),
                                              MixedScore(  40, -10), MixedScore( 41, -15), MixedScore(42, -20), 
                                              MixedScore(  43, -25) };

    constexpr MixedScore king_attacks[5] = { MixedScore(0, 0), MixedScore(10, 5), MixedScore(50, 0), MixedScore(100, 0), MixedScore(200, 0) };
    constexpr MixedScore king_slider_attacks[4] = { MixedScore(-150, -100), MixedScore(-50, -20), MixedScore(-15, -2), MixedScore(0, 0) };

    Bitboard occupancy = board.get_pieces();

    Bitboard white_king = board.get_pieces<WHITE, KING>();
    Bitboard black_king = board.get_pieces<BLACK, KING>();

    Bitboard white_mask = eval.king_zone[WHITE] | white_king;
    Bitboard black_mask = eval.king_zone[BLACK] | black_king;

    Bitboard white_pawns = board.get_pieces<WHITE, PAWN>();
    Bitboard black_pawns = board.get_pieces<BLACK, PAWN>();


    // Pawn shelter
    eval.fields[WHITE].king_safety += pawn_shelter[((white_mask | white_mask.shift< 2 * Up>()) & white_pawns).count()];
    eval.fields[BLACK].king_safety += pawn_shelter[((black_mask | black_mask.shift<-2 * Up>()) & black_pawns).count()];


    // Back-rank bonus
    eval.fields[WHITE].king_safety += MixedScore(50, -50) * (white_king & Bitboards::rank_1).count();
    eval.fields[BLACK].king_safety += MixedScore(50, -50) * (black_king & Bitboards::rank_8).count();


    // X-rays with enemy sliders?
    Square white_king_sq = white_king.bitscan_forward();
    Square black_king_sq = black_king.bitscan_forward();
    Bitboard white_rooks   = board.get_pieces<WHITE, ROOK>()   | board.get_pieces<WHITE, QUEEN>();
    Bitboard white_bishops = board.get_pieces<WHITE, BISHOP>() | board.get_pieces<WHITE, QUEEN>();
    Bitboard black_rooks   = board.get_pieces<BLACK, ROOK>()   | board.get_pieces<BLACK, QUEEN>();
    Bitboard black_bishops = board.get_pieces<BLACK, BISHOP>() | board.get_pieces<BLACK, QUEEN>();
    Bitboard wsa = (Bitboards::ranks_files[white_king.bitscan_forward()] & black_rooks)
                 | (Bitboards::diagonals  [white_king.bitscan_forward()] & black_bishops);
    Bitboard bsa = (Bitboards::ranks_files[black_king.bitscan_forward()] & white_rooks)
                 | (Bitboards::diagonals  [black_king.bitscan_forward()] & white_bishops);
    while (wsa)
        eval.fields[WHITE].king_safety += king_slider_attacks[std::min(3, Bitboards::between(white_king_sq, wsa.bitscan_forward_reset()).count())];
    while (bsa)
        eval.fields[BLACK].king_safety += king_slider_attacks[std::min(3, Bitboards::between(black_king_sq, bsa.bitscan_forward_reset()).count())];


    // Attackers to the squares near the king
    eval.fields[WHITE].king_safety -= king_attacks[std::min(4, eval.king_attackers[BLACK].count())];
    eval.fields[BLACK].king_safety -= king_attacks[std::min(4, eval.king_attackers[WHITE].count())];


    // King out in the open?
    Bitboard white_slides = Bitboards::get_attacks<BISHOP>(white_king.bitscan_forward(), occupancy)
                          | Bitboards::get_attacks<  ROOK>(white_king.bitscan_forward(), occupancy);
    Bitboard black_slides = Bitboards::get_attacks<BISHOP>(black_king.bitscan_forward(), occupancy)
                          | Bitboards::get_attacks<  ROOK>(black_king.bitscan_forward(), occupancy);
    int white_possible_dirs = white_mask.count() - 1;
    int black_possible_dirs = black_mask.count() - 1;
    int white_safe_dirs = (white_slides & board.get_pieces<WHITE>()).count();
    int black_safe_dirs = (black_slides & board.get_pieces<BLACK>()).count();
    eval.fields[WHITE].king_safety += MixedScore(-15, 8) * (white_possible_dirs - white_safe_dirs);
    eval.fields[BLACK].king_safety += MixedScore(-15, 8) * (black_possible_dirs - black_safe_dirs);


    // Open or semi-open files near the king
    constexpr MixedScore semi_open_us   = MixedScore(-25, 0);
    constexpr MixedScore semi_open_them = MixedScore(-15, 0);
    constexpr MixedScore full_open      = MixedScore(-35, 0);
    constexpr MixedScore semi_rook      = MixedScore(-25, 0);
    constexpr MixedScore full_rook      = MixedScore(-75, 0);
    Bitboard king_files[2] = { eval.king_zone[WHITE].fill_file(), eval.king_zone[BLACK].fill_file()};
    Bitboard full_open_files = eval.pawns[WHITE].open_files & eval.pawns[BLACK].open_files;
    Bitboard semi_open_files[2] = { eval.pawns[WHITE].open_files & ~full_open_files,
                                    eval.pawns[BLACK].open_files & ~full_open_files };
    Bitboard candidate_rooks[2] = { (board.get_pieces<WHITE, ROOK>() | board.get_pieces<WHITE, QUEEN>()) & king_files[BLACK],
                                    (board.get_pieces<BLACK, ROOK>() | board.get_pieces<BLACK, QUEEN>()) & king_files[WHITE] };
    // Semi-open for us
    eval.fields[WHITE].king_safety += semi_open_us * (king_files[WHITE] & semi_open_files[WHITE] & Bitboards::rank_1).count();
    eval.fields[BLACK].king_safety += semi_open_us * (king_files[BLACK] & semi_open_files[BLACK] & Bitboards::rank_1).count();
    // Semi-open for them
    eval.fields[WHITE].king_safety += semi_open_them * (king_files[WHITE] & semi_open_files[BLACK] & Bitboards::rank_1).count();
    eval.fields[BLACK].king_safety += semi_open_them * (king_files[BLACK] & semi_open_files[WHITE] & Bitboards::rank_1).count();
    // Fully open
    eval.fields[WHITE].king_safety += full_open * (king_files[WHITE] & full_open_files & Bitboards::rank_1).count();
    eval.fields[BLACK].king_safety += full_open * (king_files[BLACK] & full_open_files & Bitboards::rank_1).count();
    // Semi-open for them with rooks
    eval.fields[WHITE].king_safety += semi_rook * (king_files[WHITE] & semi_open_files[BLACK] & candidate_rooks[BLACK]).count();
    eval.fields[BLACK].king_safety += semi_rook * (king_files[BLACK] & semi_open_files[WHITE] & candidate_rooks[WHITE]).count();
    // Fully open with rooks
    eval.fields[WHITE].king_safety += full_rook * (king_files[WHITE] & full_open_files & candidate_rooks[BLACK]).count();
    eval.fields[BLACK].king_safety += full_rook * (king_files[BLACK] & full_open_files & candidate_rooks[WHITE]).count();


    return eval.fields[WHITE].king_safety - eval.fields[BLACK].king_safety;
}


Score evaluation(const Position& position, bool output)
{
    auto board = position.board();
    MixedScore mixed_result(0, 0);

    //// Check for specialised endgame evaluations
    //EvalFunc endgame_func = board.phase() < 25 ? Endgame::specialised_eval(position) : nullptr;
    //Score endgame_score = endgame_func ? endgame_func(position) : SCORE_NONE;
    //if (endgame_score != SCORE_NONE)
    //    return endgame_score;

    // Initialise helpers
    EvalData eval;
    eval.king_zone[WHITE] = Bitboards::get_attacks<KING>(board.get_pieces<WHITE, KING>().bitscan_forward(), Bitboard());
    eval.king_zone[BLACK] = Bitboards::get_attacks<KING>(board.get_pieces<BLACK, KING>().bitscan_forward(), Bitboard());

    // Material and PSQT: incrementally updated in the position (with eg scaling)
    mixed_result += board.material_eval() / MixedScore(10, 5);

    // Pawn structure
    mixed_result += pawn_structure(board, eval);

    // Mobility
    mixed_result += pieces(board, eval);

    // King safety
    mixed_result += king_safety(board, eval);

    // Tapered eval
    Score result = mixed_result.tapered(board.phase());

    // We don't return exact draw scores -> add one centipawn to the moving side
    if (result == SCORE_DRAW)
        result += turn_to_color(board.turn());

    // Cap winning scores
    result = result >=  SCORE_WIN ? SCORE_WIN - 1 :
             result <= -SCORE_WIN ? 1 - SCORE_WIN :
             result;

    // Output results
    if (output)
    {
        if (position.is_check())
        {
            std::cout << "No eval: in check" << std::endl;
            return result;
        }

        // Recompute material terms
        material(board, eval);
        piece_square_value(board, eval);

        // Display the evaluation table
        std::cout << "---------------------------------------------------------------"                                       << std::endl;
        std::cout << "               |     White     |     Black     |     Total     "                                       << std::endl;
        std::cout << " Term          |   MG     EG   |   MG     EG   |   MG     EG   "                                       << std::endl;
        std::cout << "---------------------------------------------------------------"                                       << std::endl;
        std::cout << " Material      | " << Term(eval.fields[WHITE].material  / 10, eval.fields[BLACK].material  / 10, true) << std::endl;
        std::cout << " Placement     | " << Term(eval.fields[WHITE].placement / 10, eval.fields[BLACK].placement / 10, true) << std::endl;
        std::cout << " Pawns         | " << Term(eval.fields[WHITE].pawns,          eval.fields[BLACK].pawns)                << std::endl;
        std::cout << " Mobility      | " << Term(eval.fields[WHITE].mobility,       eval.fields[BLACK].mobility)             << std::endl;
        std::cout << " King safety   | " << Term(eval.fields[WHITE].king_safety,    eval.fields[BLACK].king_safety)          << std::endl;
        std::cout << "---------------------------------------------------------------"                                       << std::endl;
        std::cout << "                                         Phase |    " << std::setw(4) << (int)board.phase()            << std::endl;
        std::cout << "                                         Final | "    << std::setw(5) << result / 100.0 << " (White)"  << std::endl;
        std::cout << "---------------------------------------------------------------"                                       << std::endl;
        std::cout << std::endl;
    }
    return result;
}


std::ostream& operator<<(std::ostream& stream, MixedScore score)
{
    constexpr auto spacer = "  ";
    stream << std::setw(5) << score.middlegame() / 100.0 << spacer << std::setw(5) << score.endgame() / 100.0;
    return stream;
}


std::ostream& operator<<(std::ostream& stream, Term term)
{
    stream << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    if (term.m_skip)
        stream << " ----   ----  | ----    ----  | " << term.m_white - term.m_black;
    else
        stream << term.m_white << "  | " << term.m_black << "  | " << term.m_white - term.m_black;
    return stream;
}


void write_field(std::ostream& stream, MixedScore white, MixedScore black)
{
    constexpr auto spacer = "  ";
    MixedScore total = white - black;
    stream << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    stream << white.middlegame() / 100.0 << spacer << white.endgame() / 100.0;
    stream << spacer << "|" << spacer;
    stream << black.middlegame() / 100.0 << spacer << black.endgame() / 100.0;
    stream << spacer << "|" << spacer;
    stream << total.middlegame() / 100.0 << spacer << total.endgame() / 100.0;
}