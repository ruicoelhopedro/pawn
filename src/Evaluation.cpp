#include "Types.hpp"
#include "Bitboard.hpp"
#include "Position.hpp"
#include "Evaluation.hpp"
#include "PieceSquareTables.hpp"
#include <cassert>
#include <stdlib.h>

namespace Evaluation
{

EvalData::EvalData(const Board& board)
{
    Square kings[] = { board.get_pieces<WHITE, KING>().bitscan_forward(),
                       board.get_pieces<BLACK, KING>().bitscan_forward() };
    for (auto turn : { WHITE, BLACK })
        king_zone[turn] = Bitboards::get_attacks<KING>(kings[turn], Bitboard());
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

MixedScore pawns(const Board& board, EvalData& data)
{
    // Bonuses and penalties
    constexpr MixedScore DoubledPenalty(-13, -51);
    constexpr MixedScore IsolatedPenalty(-3, -15);
    constexpr MixedScore BackwardPenalty(-9, -22);
    constexpr MixedScore IslandPenalty(-3, -12);
    constexpr MixedScore NonPushedCentralPenalty(-35, -50);

    // Some helpers
    constexpr Direction Up = 8;
    PawnStructure& wps = data.pawns[WHITE];
    PawnStructure& bps = data.pawns[BLACK];
    const Bitboard pawns[] = { board.get_pieces<WHITE, PAWN>(), board.get_pieces<BLACK, PAWN>() };

    // Build fields required in other evaluation terms
    wps.attacks    = Bitboards::get_attacks_pawns<WHITE>(pawns[WHITE]);
    bps.attacks    = Bitboards::get_attacks_pawns<BLACK>(pawns[BLACK]);
    wps.span       = wps.attacks.fill< Up>();
    bps.span       = bps.attacks.fill<-Up>();
    wps.outposts   = pawns[WHITE].shift< Up>() & ~wps.span;
    bps.outposts   = pawns[BLACK].shift<-Up>() & ~bps.span;
    wps.open_files = ~(pawns[WHITE].fill< Up>().fill<-Up>());
    bps.open_files = ~(pawns[BLACK].fill<-Up>().fill< Up>());
    wps.passed     = pawns[WHITE] & ~(bps.span | pawns[BLACK].fill<-Up>());
    bps.passed     = pawns[BLACK] & ~(wps.span | pawns[WHITE].fill< Up>());

    // Isolated pawns
    Bitboard isolated[] = { pawns[WHITE] & Bitboards::isolated_mask(wps.open_files),
                            pawns[BLACK] & Bitboards::isolated_mask(bps.open_files) };

    // Doubled pawns (excluding the frontmost doubled pawn)
    Bitboard doubled[] = { pawns[WHITE] & pawns[WHITE].fill_excluded<-Up>(),
                           pawns[BLACK] & pawns[BLACK].fill_excluded< Up>() };

    // Backward pawns (not defended and cannot safely push)
    Bitboard backward[] = { pawns[WHITE] & (~wps.span.shift<-Up>() & bps.attacks).shift<-Up>(),
                            pawns[BLACK] & (~bps.span.shift< Up>() & wps.attacks).shift< Up>() };

    // Build pawn structure scores
    for (auto turn : { WHITE, BLACK })
    {
        // Basic penalties
        data.fields[turn].pieces[PAWN] = DoubledPenalty  * doubled[turn].count()
                                       + IsolatedPenalty * isolated[turn].count()
                                       + BackwardPenalty * backward[turn].count()
                                       + IslandPenalty   * Bitboards::file_count(data.pawns[turn].open_files);
        
        // Update attack tables
        data.attacks[turn].push<PAWN>(data.pawns[turn].attacks);
    }

    return data.fields[WHITE].pieces[PAWN] - data.fields[BLACK].pieces[PAWN];
}


template <PieceType PIECE, Turn TURN>
MixedScore piece(const Board& board, Bitboard occupancy, EvalData& data)
{
    // Various bonuses and penalties
    constexpr MixedScore RooksConnected(15, 5);
    constexpr MixedScore RookOn7th(10, 15);
    constexpr MixedScore RookOnOpen(20, 7);
    constexpr MixedScore BehindEnemyLines(5, 4);
    constexpr MixedScore SafeBehindEnemyLines(25, 10);
    constexpr MixedScore BishopPair(10, 20);
    constexpr MixedScore DefendedByPawn(5, 1);

    // Mobility scores
    constexpr MixedScore BonusPerMove(10, 10);
    constexpr MixedScore NominalMoves = PIECE == KNIGHT ? MixedScore(4, 4)
                                      : PIECE == BISHOP ? MixedScore(5, 5)
                                      : PIECE == ROOK   ? MixedScore(4, 6)
                                      :                   MixedScore(7, 9); // QUEEN

    MixedScore score(0, 0);

    Bitboard b = board.get_pieces<TURN, PIECE>();

    // Ignore friendly sliders
    if (PIECE == BISHOP || PIECE == QUEEN)
        occupancy &= ~(board.get_pieces<TURN, BISHOP>() | board.get_pieces<TURN, QUEEN>());
    if (PIECE == ROOK || PIECE == QUEEN)
        occupancy &= ~(board.get_pieces<TURN, ROOK>() | board.get_pieces<TURN, QUEEN>());

    while (b)
    {
        Square square = b.bitscan_forward_reset();
        Bitboard attacks = Bitboards::get_attacks<PIECE>(square, occupancy);
        data.attacks[TURN].push<PIECE>(attacks);

        if (attacks & data.king_zone[~TURN])
            data.king_attackers[~TURN].set(square);

        int safe_squares = (attacks & ~data.attacks[~TURN].get_less_valuable<PIECE>()).count();
        score += (MixedScore(safe_squares, safe_squares) - NominalMoves) * BonusPerMove;

        // TODO: other terms
        if (PIECE == KNIGHT)
        {

        }
        else if (PIECE == BISHOP)
        {

        }
        else if (PIECE == ROOK)
        {
            // Connects to another rook?
            score += RooksConnected * (b & attacks).count();
        }
        else if (PIECE == QUEEN)
        {

        }
    }

    // General placement terms
    b = board.get_pieces<TURN, PIECE>();
    // Behind enemy lines?
    score += BehindEnemyLines * NominalMoves * (b & ~data.pawns[~TURN].span).count();

    // Set-wise terms
    if (PIECE == KNIGHT)
    {
        // Defended by pawns?
        score += DefendedByPawn * (b & data.pawns[TURN].attacks).count();
        // Additional bonus if behind enemy lines and defended by pawns
        score += SafeBehindEnemyLines * (b & ~data.pawns[~TURN].span & data.pawns[TURN].attacks).count();
    }
    else if (PIECE == BISHOP)
    {
        // Defended by pawns?
        score += DefendedByPawn * (b & data.pawns[TURN].attacks).count();
        // Additional bonus if behind enemy lines and defended by pawns
        score += SafeBehindEnemyLines * (b & ~data.pawns[~TURN].span & data.pawns[TURN].attacks).count();

        // Bishop pair?
        if ((b & Bitboards::square_color[WHITE]) && (b & Bitboards::square_color[BLACK]))
            score += BishopPair;
    }
    else if (PIECE == ROOK)
    {
        // Rooks on 7th rank?
        constexpr Bitboard rank7 = TURN == WHITE ? Bitboards::rank_7 : Bitboards::rank_2;
        score += RookOn7th * (b & rank7).count();
        // Files for each rook: check if open or semi-open
        score += RookOnOpen * (b & data.pawns[TURN].open_files).count();
    }
    else if (PIECE == QUEEN)
    {

    }

    data.fields[TURN].pieces[PIECE] = score;
    return score;
}


MixedScore pieces(const Board& board, EvalData& data)
{
    MixedScore result(0, 0);
    Bitboard occupancy = board.get_pieces<WHITE>() | board.get_pieces<BLACK>();

    result += piece<KNIGHT, WHITE>(board, occupancy, data)
            - piece<KNIGHT, BLACK>(board, occupancy, data);
    result += piece<BISHOP, WHITE>(board, occupancy, data)
            - piece<BISHOP, BLACK>(board, occupancy, data);
    result += piece<  ROOK, WHITE>(board, occupancy, data)
            - piece<  ROOK, BLACK>(board, occupancy, data);
    result += piece< QUEEN, WHITE>(board, occupancy, data)
            - piece< QUEEN, BLACK>(board, occupancy, data);

    return result;
}


template<Turn TURN>
MixedScore king_safety(const Board& board, EvalData& data)
{
    constexpr Direction Up = (TURN == WHITE) ? 8 : -8;
    constexpr Bitboard Rank1 = (TURN == WHITE) ? Bitboards::rank_1 : Bitboards::rank_8;

    constexpr MixedScore BackRankBonus(50, -50);
    constexpr MixedScore OpenRay(-15, 8);
    constexpr MixedScore PawnShelter[] = { MixedScore(-100,   0), MixedScore(-25,   0), MixedScore( 0,   0),
                                           MixedScore(  25,   0), MixedScore( 35,  -5), MixedScore(40,  -5),
                                           MixedScore(  40, -10), MixedScore( 41, -15), MixedScore(42, -20) };

    constexpr MixedScore SliderAttackers[] = { MixedScore(-150, -100), MixedScore(-50, -20),
                                               MixedScore( -15,   -2), MixedScore(  0,   0),
                                               MixedScore(   0,    0), MixedScore(  0,   0),
                                               MixedScore(   0,    0) };

    Bitboard occupancy = board.get_pieces();

    const Bitboard king_bb = board.get_pieces<TURN, KING>();
    const Bitboard pawns_bb = board.get_pieces<TURN, PAWN>();
    const Square king_sq = king_bb.bitscan_forward();
    const Bitboard mask = Bitboards::get_attacks<KING>(king_sq, occupancy) | king_bb;

    MixedScore score(0, 0);

    // Pawn shelter
    Bitboard shelter_zone = mask | mask.shift<2*Up>();
    score += PawnShelter[(pawns_bb & shelter_zone).count()];

    // Back-rank bonus
    score += BackRankBonus * Rank1.test(king_sq);

    // X-rays with enemy sliders
    Bitboard their_rooks   = board.get_pieces<~TURN,   ROOK>() | board.get_pieces<~TURN, QUEEN>();
    Bitboard their_bishops = board.get_pieces<~TURN, BISHOP>() | board.get_pieces<~TURN, QUEEN>();
    Bitboard slider_attackers = (Bitboards::ranks_files[king_sq] & their_rooks)
                              | (Bitboards::diagonals[king_sq]   & their_bishops);
    while (slider_attackers)
        score += SliderAttackers[Bitboards::between(king_sq, slider_attackers.bitscan_forward_reset()).count()];

    // Attackers to the squares near the king
    int attacked_squares = 0;
    Bitboard b = mask;
    while(b)
        attacked_squares += board.attackers_battery<~TURN>(b.bitscan_forward_reset(), occupancy).count();
    score += MixedScore(-10, 0) * std::min(9, attacked_squares) * std::min(9, attacked_squares);

    // Possible checkers
    Bitboard checkers = (Bitboards::get_attacks_pawns<TURN>(king_sq)        &  data.attacks[~TURN].get<PAWN>())
                      | (Bitboards::get_attacks<KNIGHT>(king_sq, occupancy) &  data.attacks[~TURN].get<KNIGHT>())
                      | (Bitboards::get_attacks<BISHOP>(king_sq, occupancy) & (data.attacks[~TURN].get<BISHOP>() | data.attacks[~TURN].get<QUEEN>()))
                      | (Bitboards::get_attacks<  ROOK>(king_sq, occupancy) & (data.attacks[~TURN].get<  ROOK>() | data.attacks[~TURN].get<QUEEN>()));
    score += MixedScore(-75, 0) * (checkers & ~data.attacks[TURN].get()).count();

    // King out in the open
    Bitboard rays = Bitboards::get_attacks<BISHOP>(king_sq, occupancy)
                  | Bitboards::get_attacks<  ROOK>(king_sq, occupancy);
    int safe_dirs = (rays & board.get_pieces<TURN>()).count();
    score += OpenRay * std::max(0, mask.count() - safe_dirs - 3);

    data.fields[TURN].pieces[KING] = score;
    return score;
}


template<Turn TURN>
MixedScore space(const Board& board, EvalData& data)
{
    constexpr MixedScore CenterSquareControl(15, 1);

    // Center control
    Bitboard center = Bitboards::zone1 | Bitboards::zone2;
    Bitboard control_bb = data.attacks[TURN].get() & ~data.attacks[~TURN].get();

    data.fields[TURN].space = CenterSquareControl * (control_bb & center).count();
    return data.fields[TURN].space;
}


template<Turn TURN>
MixedScore passed(const Board& board, EvalData& data)
{
    constexpr Direction Up = (TURN == WHITE) ? 8 : -8;

    constexpr MixedScore PassedBonus[] = { MixedScore(  0,   0), MixedScore(  0,   0),
                                           MixedScore(  7,  27), MixedScore( 16,  32),
                                           MixedScore( 17,  40), MixedScore( 64,  71),
                                           MixedScore(170, 174), MixedScore(278, 262) };

    MixedScore score(0, 0);

    // Score each passed pawn
    Bitboard passed = data.pawns[TURN].passed;
    while (passed)
    {
        Square p = passed.bitscan_forward_reset();

        // Basic score based on rank
        int r = rank(p, TURN);
        score += PassedBonus[r];

        // More bonuses if the pawn can push forward and is above 3rd rank
        Square block = p + Up;
        if (!board.get_pieces().test(block) && r >= 3)
        {
            int bonus = 0;
            int weight = std::max(0, 5 * r - 10);
            Bitboard span = Bitboard::from_single_bit(p).fill_excluded<Up>();

            // No enemies in the pawn span?
            if (!(board.get_pieces<~TURN>() & span))
                bonus += 10;
            
            // Enemy does not attack pawn span?
            if (!(data.attacks[~TURN].get() & span))
                bonus += 20;
            
            // Otherwise, check if we can safely push
            else if (!data.attacks[~TURN].get().test(block))
                bonus += 5;

            score += MixedScore(bonus * weight, bonus * weight);
        }
    }

    data.fields[TURN].passed = score;
    return score;
}


template<Turn TURN>
MixedScore threats(const Board& board, EvalData& data)
{
    constexpr MixedScore Hanging(75, 75);

    MixedScore mixed_result(0, 0);

    Bitboard opp_pieces = board.get_pieces<~TURN>() & ~board.get_pieces<~TURN, PAWN>();
    Bitboard controlled = data.attacks[TURN].get() & ~data.attacks[~TURN].get();

    mixed_result += Hanging * (opp_pieces & controlled).count();

    data.fields[TURN].threats = mixed_result;
    return mixed_result;
}


Score evaluation(const Board& board, EvalData& data)
{
    MixedScore mixed_result(0, 0);

    // Material and PSQT: incrementally updated in the position (with eg scaling)
    mixed_result += board.material_eval() * MixedScore(1, 2);

    // Pawn structure
    mixed_result += pawns(board, data);

    // Piece scores
    mixed_result += pieces(board, data);

    // King safety
    mixed_result += king_safety<WHITE>(board, data) - king_safety<BLACK>(board, data);

    // Space
    mixed_result += space<WHITE>(board, data) - space<BLACK>(board, data);

    // Threats
    mixed_result += threats<WHITE>(board, data) - threats<BLACK>(board, data);

    // Passed pawn scores
    mixed_result += passed<WHITE>(board, data) - passed<BLACK>(board, data);

    // Tapered eval
    Score result = mixed_result.tapered(board.phase());

    // We don't return exact draw scores -> add one centipawn to the moving side
    if (result == SCORE_DRAW)
        result += turn_to_color(board.turn());

    return result;
}


void eval_table(const Board& board, EvalData& data, Score score)
{
    // No eval printing when in check
    if (board.checkers())
    {
        std::cout << "No eval: in check" << std::endl;
        return;
    }

    // Update material and placement terms
    material(board, data);
    piece_square_value(board, data);

    // Print the eval table
    std::cout << "---------------------------------------------------------------"                                        << std::endl;
    std::cout << "               |     White     |     Black     |     Total     "                                        << std::endl;
    std::cout << " Term          |   MG     EG   |   MG     EG   |   MG     EG   "                                        << std::endl;
    std::cout << "---------------------------------------------------------------"                                        << std::endl;
    std::cout << " Material      | " << Term< true>(data.fields[WHITE].material,       data.fields[BLACK].material)       << std::endl;
    std::cout << " Placement     | " << Term< true>(data.fields[WHITE].placement / 10, data.fields[BLACK].placement / 10) << std::endl;
    std::cout << " Pawns         | " << Term<false>(data.fields[WHITE].pieces[PAWN],   data.fields[BLACK].pieces[PAWN])   << std::endl;
    std::cout << " Knights       | " << Term<false>(data.fields[WHITE].pieces[KNIGHT], data.fields[BLACK].pieces[KNIGHT]) << std::endl;
    std::cout << " Bishops       | " << Term<false>(data.fields[WHITE].pieces[BISHOP], data.fields[BLACK].pieces[BISHOP]) << std::endl;
    std::cout << " Rooks         | " << Term<false>(data.fields[WHITE].pieces[ROOK],   data.fields[BLACK].pieces[ROOK])   << std::endl;
    std::cout << " Queens        | " << Term<false>(data.fields[WHITE].pieces[QUEEN],  data.fields[BLACK].pieces[QUEEN])  << std::endl;
    std::cout << " King safety   | " << Term<false>(data.fields[WHITE].pieces[KING],   data.fields[BLACK].pieces[KING])   << std::endl;
    std::cout << " Space         | " << Term<false>(data.fields[WHITE].space,          data.fields[BLACK].space)          << std::endl;
    std::cout << " Threats       | " << Term<false>(data.fields[WHITE].threats,        data.fields[BLACK].threats)        << std::endl;
    std::cout << " Passed pawns  | " << Term<false>(data.fields[WHITE].passed,         data.fields[BLACK].passed)         << std::endl;
    std::cout << "---------------------------------------------------------------"                                        << std::endl;
    std::cout << "                                         Phase |    " << std::setw(4) << (int)board.phase()             << std::endl;
    std::cout << "                                         Final | "    << std::setw(5) << score / 100.0 << " (White)"    << std::endl;
    std::cout << "---------------------------------------------------------------"                                        << std::endl;
    std::cout << std::endl;
}
    
}