#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include <algorithm>


using Square = int8_t;
using Value = int8_t;
using Hash = uint64_t;
using Piece = int8_t;
using Direction = int8_t;
using Score = int16_t;
using Depth = uint8_t;


constexpr int NUM_COLORS = 2;
constexpr int NUM_SQUARES = 64;
constexpr int NUM_PIECE_TYPES = 6;
constexpr int NUM_CASTLE_SIDES = 2;


constexpr int NUM_MAX_MOVES = 256;
constexpr Depth NUM_MAX_DEPTH = 200;


enum Turn : int8_t
{
    WHITE = 0,
    BLACK = 1
};


enum Color : int8_t
{
    WHITE_COLOR = 1,
    BLACK_COLOR = -1
};


enum PieceType : int8_t
{
    PIECE_NONE = -1,
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5
};


enum BoardPieces : int8_t
{
    NO_PIECE = 2 * PIECE_NONE,
    W_PAWN = 2 * PAWN,
    W_KNIGHT = 2 * KNIGHT,
    W_BISHOP = 2 * BISHOP,
    W_ROOK = 2 * ROOK,
    W_QUEEN = 2 * QUEEN,
    W_KING = 2 * KING,
    B_PAWN = W_PAWN + 1,
    B_KNIGHT = W_KNIGHT + 1,
    B_BISHOP = W_BISHOP + 1,
    B_ROOK = W_ROOK + 1,
    B_QUEEN = W_QUEEN + 1,
    B_KING = W_KING + 1
};


enum MoveType : int8_t
{
    INVALID_MOVE = -1,
    QUIET = 0,
    DOUBLE_PAWN_PUSH = 1,
    KING_CASTLE = 2,
    QUEEN_CASTLE = 3,
    CAPTURE = 4,
    EP_CAPTURE = 5,
    INVALID_1 = 6,
    INVALID_2 = 7,
    KNIGHT_PROMO = 8,
    BISHOP_PROMO = 9,
    ROOK_PROMO = 10,
    QUEEN_PROMO = 11,
    KNIGHT_PROMO_CAPTURE = 12,
    BISHOP_PROMO_CAPTURE = 13,
    ROOK_PROMO_CAPTURE = 14,
    QUEEN_PROMO_CAPTURE = 15
};


enum Squares : int8_t
{
    SQUARE_NULL = -1,
    SQUARE_A1, SQUARE_B1, SQUARE_C1, SQUARE_D1, SQUARE_E1, SQUARE_F1, SQUARE_G1, SQUARE_H1,
    SQUARE_A2, SQUARE_B2, SQUARE_C2, SQUARE_D2, SQUARE_E2, SQUARE_F2, SQUARE_G2, SQUARE_H2,
    SQUARE_A3, SQUARE_B3, SQUARE_C3, SQUARE_D3, SQUARE_E3, SQUARE_F3, SQUARE_G3, SQUARE_H3,
    SQUARE_A4, SQUARE_B4, SQUARE_C4, SQUARE_D4, SQUARE_E4, SQUARE_F4, SQUARE_G4, SQUARE_H4,
    SQUARE_A5, SQUARE_B5, SQUARE_C5, SQUARE_D5, SQUARE_E5, SQUARE_F5, SQUARE_G5, SQUARE_H5,
    SQUARE_A6, SQUARE_B6, SQUARE_C6, SQUARE_D6, SQUARE_E6, SQUARE_F6, SQUARE_G6, SQUARE_H6,
    SQUARE_A7, SQUARE_B7, SQUARE_C7, SQUARE_D7, SQUARE_E7, SQUARE_F7, SQUARE_G7, SQUARE_H7,
    SQUARE_A8, SQUARE_B8, SQUARE_C8, SQUARE_D8, SQUARE_E8, SQUARE_F8, SQUARE_G8, SQUARE_H8
};


enum CastleSide
{
    KINGSIDE = 0,
    QUEENSIDE = 1
};


enum ScoreType : Score
{
    SCORE_ZERO = 0,
    SCORE_DRAW = 0,
    SCORE_MATE = 32000,
    SCORE_MATE_FOUND = SCORE_MATE - NUM_MAX_DEPTH - 1,
    SCORE_INFINITE = 32001,
    SCORE_NONE = 32002
};


namespace Phases
{
    constexpr int Pawn = 0;
    constexpr int Knight = 1;
    constexpr int Bishop = 1;
    constexpr int Rook = 2;
    constexpr int Queen = 4;
    constexpr int King = 0;
    constexpr int Pieces[NUM_PIECE_TYPES] = { Pawn, Knight, Bishop, Rook, Queen, King };
    constexpr int Total = 16 * Pawn + 4 * Knight + 4 * Bishop + 4 * Rook + 2 * Queen + 2 * King;
}


class MixedScore
{
    Score mg;
    Score eg;

public:
    inline constexpr MixedScore(Score smg, Score seg) : mg(smg), eg(seg) {}

    inline constexpr Score middlegame() const { return mg; }
    inline constexpr Score endgame() const { return eg; }

    inline constexpr Score tapered(uint8_t phase_entry) const
    {
        int phase = (phase_entry * 256 + (Phases::Total / 2)) / Phases::Total;
        return ((mg * (256 - phase)) + (eg * phase)) / 256;
    }

    inline constexpr MixedScore& operator+=(const MixedScore& other) { mg += other.mg; eg += other.eg; return *this; }
    inline constexpr MixedScore& operator-=(const MixedScore& other) { mg -= other.mg; eg -= other.eg; return *this; }
    inline constexpr MixedScore& operator*=(const MixedScore& other) { mg *= other.mg; eg *= other.eg; return *this; }
    inline constexpr MixedScore& operator/=(const MixedScore& other) { mg /= other.mg; eg /= other.eg; return *this; }

    inline constexpr MixedScore& operator+=(const Score& other) { mg += other; eg += other; return *this; }
    inline constexpr MixedScore& operator-=(const Score& other) { mg -= other; eg -= other; return *this; }
    inline constexpr MixedScore& operator*=(const Score& other) { mg *= other; eg *= other; return *this; }
    inline constexpr MixedScore& operator/=(const Score& other) { mg /= other; eg /= other; return *this; }

    inline constexpr MixedScore& operator+=(int other) { mg += other; eg += other; return *this; }
    inline constexpr MixedScore& operator-=(int other) { mg -= other; eg -= other; return *this; }
    inline constexpr MixedScore& operator*=(int other) { mg *= other; eg *= other; return *this; }
    inline constexpr MixedScore& operator/=(int other) { mg /= other; eg /= other; return *this; }

    inline constexpr MixedScore operator+(const MixedScore& other) const { return MixedScore(mg + other.mg, eg + other.eg); }
    inline constexpr MixedScore operator-(const MixedScore& other) const { return MixedScore(mg - other.mg, eg - other.eg); }
    inline constexpr MixedScore operator*(const MixedScore& other) const { return MixedScore(mg * other.mg, eg * other.eg); }
    inline constexpr MixedScore operator/(const MixedScore& other) const { return MixedScore(mg / other.mg, eg / other.eg); }

    inline constexpr MixedScore operator+(const Score& other) const { return MixedScore(mg + other, eg + other); }
    inline constexpr MixedScore operator-(const Score& other) const { return MixedScore(mg - other, eg - other); }
    inline constexpr MixedScore operator*(const Score& other) const { return MixedScore(mg * other, eg * other); }
    inline constexpr MixedScore operator/(const Score& other) const { return MixedScore(mg / other, eg / other); }

    inline constexpr MixedScore operator+(int other) const { return MixedScore(mg + other, eg + other); }
    inline constexpr MixedScore operator-(int other) const { return MixedScore(mg - other, eg - other); }
    inline constexpr MixedScore operator*(int other) const { return MixedScore(mg * other, eg * other); }
    inline constexpr MixedScore operator/(int other) const { return MixedScore(mg / other, eg / other); }
};


constexpr uint64_t uint64(int i) { return static_cast<uint64_t>(i); }


constexpr bool inside_board(int file, int rank) { return (file >= 0 && file < 8 && rank >= 0 && rank < 8); }


constexpr int rank(Square square) { return square / 8; }
constexpr int file(Square square) { return square % 8; }
constexpr Square make_square(int rank, int file) { return file + 8 * rank; }


constexpr Color turn_to_color(Turn turn) { return (turn == WHITE) ? WHITE_COLOR : BLACK_COLOR; }
constexpr Turn color_to_turn(Color color) { return (color == WHITE_COLOR) ? WHITE : BLACK; }


constexpr Color operator~(const Color& other) { return (other == WHITE_COLOR) ? BLACK_COLOR : WHITE_COLOR; }
constexpr Turn operator~(const Turn& other) { return (other == WHITE) ? BLACK : WHITE; }


constexpr Color operator-(const Color& other) { return static_cast<Color>(-static_cast<int>(other)); }


constexpr BoardPieces get_board_piece(Piece piece, Turn turn) { return static_cast<BoardPieces>(2 * piece + turn); }


constexpr bool is_mate(const Score& score)
{
    Score abs_score = (score >= 0 ? score : -score);
    return abs_score >= SCORE_MATE_FOUND && abs_score <= SCORE_MATE;
}


constexpr Score update_score(const Score& score, int depth = 1)
{
    // Non-mate score
    if (!is_mate(score))
        return score;

    // Mate score -> subtract one depth in the correct side
    return (score > 0) ? score - depth : score + depth;
}


constexpr Score score_to_tt(const Score& score, int ply)
{
    if (!is_mate(score))
        return score;

    return (score > 0) ? (score + ply) : (score - ply);
}


constexpr Score score_from_tt(const Score& score, int ply)
{
    if (!is_mate(score))
        return score;

    return (score > 0) ? (score - ply) : (score + ply);
}


constexpr int mate_in(const Score& score)
{
    int distance = (score > 0) ? (SCORE_MATE - score) : (SCORE_MATE + score);
    return (score > 0) ? (distance / 2 + 1) : (-distance / 2);
}


inline constexpr Square horizontal_distance(Square square)
{
    int this_file = file(square);
    return std::min(this_file, 7 - this_file);
}


inline constexpr Square vertical_mirror(Square square)
{
    return file(square) + (7 - rank(square));
}


std::string get_square(Square square);


std::vector<std::string> split(const std::string& s, char delimiter);
