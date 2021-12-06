#include "Types.hpp"
#include "Position.hpp"
#include "Endgame.hpp"



namespace Endgame
{
    EvalFunc specialised_eval(const Position& pos)
    {
        const Turn Us = pos.get_turn();

        if (pos.board().non_pawn_material())
        {
            Bitboard white_pieces = pos.board().get_pieces<WHITE>() ^ pos.board().get_pieces<WHITE, KING>();
            Bitboard black_pieces = pos.board().get_pieces<BLACK>() ^ pos.board().get_pieces<BLACK, KING>();

            // One of the sides has a bare king?
            if (!white_pieces || !black_pieces)
            {
                // The side with a queen or rook is always winning
                Bitboard white_rq = pos.board().get_pieces<WHITE, ROOK>() | pos.board().get_pieces<WHITE, QUEEN>();
                Bitboard black_rq = pos.board().get_pieces<BLACK, ROOK>() | pos.board().get_pieces<BLACK, QUEEN>();
                if (white_rq | black_rq)
                    return white_rq ? Endgame::KingToCorner<ANY_CORNER, 1000, WHITE>
                                    : Endgame::KingToCorner<ANY_CORNER, 1000, BLACK>;

                // If no rooks or queens and single-piece it is a draw (KNK or KBK)
                int n_pieces = (white_pieces | black_pieces).count();
                if (n_pieces == 1)
                    return Endgame::Drawing;

                // Two-piece (knight or bishop) endgames
                Bitboard knights = pos.board().get_pieces<WHITE, KNIGHT>() | pos.board().get_pieces<BLACK, KNIGHT>();
                Bitboard bishops = pos.board().get_pieces<WHITE, BISHOP>() | pos.board().get_pieces<BLACK, BISHOP>();
                if (n_pieces == 2)
                {
                    // Knights only is drawish
                    if (!bishops)
                        return Endgame::Drawing;

                    // Bishops only
                    if (!knights)
                    {
                        // Winnable if opposite coloured bishops
                        if ((bishops & Bitboards::square_color[WHITE]) && (bishops & Bitboards::square_color[BLACK]))
                            return (white_pieces & bishops) ? Endgame::KingToCorner<ANY_CORNER, 600, WHITE>
                                                            : Endgame::KingToCorner<ANY_CORNER, 600, BLACK>;

                        // Same coloured bishops is a draw
                        return Endgame::Drawing;
                    }

                    // KNBK endgame: winnable by dragging the weak king to a corner of the same colour of our bishop
                    if (bishops & Bitboards::square_color[WHITE])
                        return (white_pieces & bishops) ? Endgame::KingToCorner<WHITE_SQUARES, 600, WHITE>
                                                        : Endgame::KingToCorner<WHITE_SQUARES, 600, BLACK>;
                    else
                        return (white_pieces & bishops) ? Endgame::KingToCorner<BLACK_SQUARES, 600, WHITE>
                                                        : Endgame::KingToCorner<BLACK_SQUARES, 600, BLACK>;
                }

                // Three or more piece endgames are generally winnable, except for all bishops of the same colour
                if (!knights && !((bishops & Bitboards::square_color[WHITE]) && (bishops & Bitboards::square_color[BLACK])))
                    return Endgame::Drawing;
                else
                    return (white_pieces & (bishops | knights)) ? Endgame::KingToCorner<ANY_CORNER, 600, WHITE>
                                                                : Endgame::KingToCorner<ANY_CORNER, 600, BLACK>;
            }
            else
            {
                Bitboard pawns   = pos.board().get_pieces<WHITE,   PAWN>() | pos.board().get_pieces<BLACK,   PAWN>();
                Bitboard knights = pos.board().get_pieces<WHITE, KNIGHT>() | pos.board().get_pieces<BLACK, KNIGHT>();
                Bitboard bishops = pos.board().get_pieces<WHITE, BISHOP>() | pos.board().get_pieces<BLACK, BISHOP>();
                Bitboard rooks   = pos.board().get_pieces<WHITE,   ROOK>() | pos.board().get_pieces<BLACK,   ROOK>();
                Bitboard queens  = pos.board().get_pieces<WHITE,  QUEEN>() | pos.board().get_pieces<BLACK,  QUEEN>();

                int n_pieces = (white_pieces | black_pieces).count();
                if (n_pieces == 2)
                {
                    if (queens && pawns)
                        return Endgame::KQKP;

                    if (rooks && pawns)
                        return Endgame::KRKP;
                }
            }
        }
        else
        {
            // Only pawns on the board
            Bitboard our_pawns = pos.board().get_pieces(Us, PAWN);
            Bitboard their_pawns = pos.board().get_pieces(~Us, PAWN);

            // KPK
            if ((our_pawns | their_pawns).count() == 1)
                return Endgame::KPK;
        }

        return nullptr;
    }


    Score Drawing(const Position& pos)
    {
        (void)pos;
        return SCORE_DRAW;
    }


    Score KPK(const Position& pos)
    {
        const Turn   moving_side = pos.get_turn();
        const Turn   strong_side = pos.board().get_pieces<WHITE, PAWN>() ? WHITE : BLACK;
        const Square pawn        = pos.board().get_pieces( strong_side, PAWN).bitscan_forward();
        const Square strong_king = pos.board().get_pieces( strong_side, KING).bitscan_forward();
        const Square weak_king   = pos.board().get_pieces(~strong_side, KING).bitscan_forward();
        const Square promoting_square = file(pawn) + (strong_side == WHITE ? SQUARE_A8 : SQUARE_A1);

        const Score winning = turn_to_color(strong_side) * (SCORE_WIN + 110 - distance(pawn, promoting_square));

        // Weak king outside the square?
        if (std::min(5, distance(pawn, promoting_square)) < distance(weak_king, promoting_square) - (moving_side != strong_side))
            return winning;

        // Pawn path protected by the strong king?
        Bitboard pawn_path = strong_side == WHITE
                           ? Bitboard::from_square(pawn).fill_excluded< 8>()
                           : Bitboard::from_square(pawn).fill_excluded<-8>();
        Bitboard strong_king_attacks = Bitboards::get_attacks<KING>(strong_king, pos.board().get_pieces());
        if (strong_king_attacks & pawn_path)
            return winning;

        // Pawn capturable?
        Bitboard weak_king_attacks = Bitboards::get_attacks<KING>(weak_king, pos.board().get_pieces());
        if (weak_king_attacks.test(pawn) && !strong_king_attacks.test(pawn))
            return SCORE_DRAW;

        // Cannot infer the result
        return SCORE_NONE;
    }



    Score KQRxK(const Position& pos)
    {
        const Turn   moving_side = pos.get_turn();
        const Turn   strong_side = (pos.board().get_pieces<WHITE, ROOK>() | pos.board().get_pieces<WHITE, QUEEN>()) ? WHITE : BLACK;
        const Square strong_king = pos.board().get_pieces( strong_side, KING).bitscan_forward();
        const Square weak_king   = pos.board().get_pieces(~strong_side, KING).bitscan_forward();

        int weak_king_distance = std::min(7 - file(weak_king), file(weak_king))
                               + std::min(7 - rank(weak_king), rank(weak_king));

        return turn_to_color(strong_side) * (SCORE_WIN + 1000 - 10 * weak_king_distance - distance(strong_king, weak_king));
    }



    template <Corner CORNER, Score BONUS, Turn WINNING>
    Score KingToCorner(const Position& pos)
    {
        const Turn   moving_side = pos.get_turn();
        const Square strong_king = pos.board().get_pieces< WINNING, KING>().bitscan_forward();
        const Square weak_king   = pos.board().get_pieces<~WINNING, KING>().bitscan_forward();

        int weak_king_distance = CORNER == WHITE_SQUARES ? std::min(distance(weak_king, SQUARE_A8), distance(weak_king, SQUARE_H1))
                               : CORNER == BLACK_SQUARES ? std::min(distance(weak_king, SQUARE_A1), distance(weak_king, SQUARE_H8))
                                                         : std::min(std::min(distance(weak_king, SQUARE_A8), distance(weak_king, SQUARE_H1)), 
                                                                    std::min(distance(weak_king, SQUARE_A1), distance(weak_king, SQUARE_H8)));

        return turn_to_color(WINNING) * (SCORE_WIN + BONUS - 10 * weak_king_distance - distance(strong_king, weak_king));
    }



    Score KQKP(const Position& pos)
    {
        const Turn   moving_side = pos.get_turn();
        const Turn   strong_side = pos.board().get_pieces<WHITE, QUEEN>() ? WHITE : BLACK;
        const Square queen       = pos.board().get_pieces( strong_side, QUEEN).bitscan_forward();
        const Square pawn        = pos.board().get_pieces(~strong_side,  PAWN).bitscan_forward();
        const Square strong_king = pos.board().get_pieces( strong_side,  KING).bitscan_forward();
        const Square weak_king   = pos.board().get_pieces(~strong_side,  KING).bitscan_forward();
        const int pawn_rank = (strong_side == WHITE ? 7 - rank(pawn) : rank(pawn)) + 1;

        // Base score: distance between kings
        Score score = 8 - distance(strong_king, weak_king);

        // Some heuristics for a *guaranteed* win
        if (pawn_rank <= 5 ||
            distance(weak_king, pawn) > 2 ||
            (Bitboards::b_file | Bitboards::d_file | Bitboards::e_file | Bitboards::g_file).test(pawn))
        {
            score += SCORE_WIN + 10;
        }
        else
        {
            // Does the strong side control any square in front of the pawn?
            Bitboard occupancy = pos.board().get_pieces();
            Bitboard strong_attacks = Bitboards::get_attacks<QUEEN>(queen,       occupancy)
                                    | Bitboards::get_attacks< KING>(strong_king, occupancy);
            Bitboard weak_attacks   = Bitboards::get_attacks< KING>(weak_king,   occupancy);
            Bitboard pawn_path = strong_side == WHITE
                               ? Bitboard::from_square(pawn).fill<-8>()
                               : Bitboard::from_square(pawn).fill< 8>();
            if (moving_side == strong_side && (strong_attacks & pawn_path & ~weak_attacks))
                score += SCORE_WIN + 10;
        }

        return turn_to_color(strong_side) * score;
    }



    Score KRKP(const Position& pos)
    {
        const Turn   moving_side = pos.get_turn();
        const Turn   strong_side = pos.board().get_pieces<WHITE, ROOK>() ? WHITE : BLACK;
        const Square rook = pos.board().get_pieces( strong_side, ROOK).bitscan_forward();
        const Square pawn = pos.board().get_pieces(~strong_side, PAWN).bitscan_forward();
        const Square strong_king = pos.board().get_pieces(strong_side, KING).bitscan_forward();
        const Square weak_king = pos.board().get_pieces(~strong_side, KING).bitscan_forward();
        const int pawn_rank = (strong_side == WHITE ? 7 - rank(pawn) : rank(pawn)) + 1;
        const int weak_king_rank = (strong_side == WHITE ? 7 - rank(weak_king) : rank(weak_king)) + 1;

        // Base score: distance between kings
        Score score = 8 - distance(strong_king, weak_king);

        // Strong king controls any square in the pawn path?
        Bitboard occupancy = pos.board().get_pieces();
        Bitboard strong_king_attacks = Bitboards::get_attacks<KING>(strong_king, occupancy) | Bitboard::from_square(strong_king);
        Bitboard weak_king_attacks   = Bitboards::get_attacks<KING>(  weak_king, occupancy);
        Bitboard pawn_path = strong_side == WHITE
                           ? Bitboard::from_square(pawn).fill<-8>()
                           : Bitboard::from_square(pawn).fill< 8>();
        if (moving_side == strong_side && (strong_king_attacks & pawn_path & ~weak_king_attacks))
            score += SCORE_WIN + 10;
        // Advanced pawn supported by the king is drawish
        else if (pawn_rank > 4 && weak_king_rank > 4 && distance(pawn, weak_king) < 3 - (moving_side != strong_side))
            score += distance(weak_king, pawn);

        return turn_to_color(strong_side) * score;
    }
}