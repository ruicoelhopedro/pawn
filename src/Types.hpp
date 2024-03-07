#pragma once
#include <inttypes.h>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <atomic>
#include <cmath>


using Square = int8_t;
using Hash = uint64_t;
using Direction = int8_t;
using Score = int32_t;
using Depth = int;


constexpr int NUM_COLORS = 2;
constexpr int NUM_SQUARES = 64;
constexpr int NUM_PIECE_TYPES = 6;
constexpr int NUM_CASTLE_SIDES = 2;


constexpr int NUM_MAX_MOVES = 256;
constexpr int NUM_MAX_PLY = 255;
constexpr Depth NUM_MAX_DEPTH = 200;


enum Turn : int8_t
{
    WHITE = 0,
    BLACK = 1
};


enum Color : int8_t
{
    WHITE_COLOR = 1,
    NO_COLOR = 0,
    BLACK_COLOR = -1
};


enum PieceType : int8_t
{
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    INVALID_PIECE = 6,
    PIECE_NONE = 7
};


enum Piece : int8_t
{
    W_PAWN   = PAWN,
    W_KNIGHT = KNIGHT,
    W_BISHOP = BISHOP,
    W_ROOK   = ROOK,
    W_QUEEN  = QUEEN,
    W_KING   = KING,
    B_PAWN   = W_PAWN   | 0x8,
    B_KNIGHT = W_KNIGHT | 0x8,
    B_BISHOP = W_BISHOP | 0x8,
    B_ROOK   = W_ROOK   | 0x8,
    B_QUEEN  = W_QUEEN  | 0x8,
    B_KING   = W_KING   | 0x8,
    NO_PIECE = PIECE_NONE
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
    QUEENSIDE = 1,
    NO_SIDE = 2
};

enum class CastleFile
{
    NONE,
    FILE_A,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H
};


enum ScoreType : Score
{
    SCORE_ZERO = 0,
    SCORE_DRAW = 0,
    SCORE_MATE = 32000,
    SCORE_TB_WIN_FOUND = SCORE_MATE - 2 * NUM_MAX_PLY - 1,
    SCORE_MATE_FOUND = SCORE_MATE - NUM_MAX_PLY - 1,
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

    constexpr int MgLimit = 4;
    constexpr int EgLimit = 19;
}


class MixedScore
{
    Score mg;
    Score eg;

public:
    inline constexpr MixedScore() : mg(0), eg(0) {}
    inline constexpr MixedScore(Score smg, Score seg) : mg(smg), eg(seg) {}

    inline constexpr Score middlegame() const { return mg; }
    inline constexpr Score endgame() const { return eg; }

    inline constexpr Score tapered(int phase_entry) const
    {
        int phase = std::clamp(256 * (phase_entry - Phases::MgLimit) / (Phases::EgLimit - Phases::MgLimit), 0, 256);
        return ((mg * (256 - phase)) + (eg * phase)) / 256;
    }

    inline constexpr MixedScore& operator+=(const MixedScore& other) { mg += other.mg; eg += other.eg; return *this; }
    inline constexpr MixedScore& operator-=(const MixedScore& other) { mg -= other.mg; eg -= other.eg; return *this; }
    inline constexpr MixedScore& operator*=(const MixedScore& other) { mg *= other.mg; eg *= other.eg; return *this; }
    inline constexpr MixedScore& operator/=(const MixedScore& other) { mg /= other.mg; eg /= other.eg; return *this; }

    inline constexpr MixedScore& operator+=(int other) { mg += other; eg += other; return *this; }
    inline constexpr MixedScore& operator-=(int other) { mg -= other; eg -= other; return *this; }
    inline constexpr MixedScore& operator*=(int other) { mg *= other; eg *= other; return *this; }
    inline constexpr MixedScore& operator/=(int other) { mg /= other; eg /= other; return *this; }

    inline constexpr MixedScore operator+(const MixedScore& other) const { return MixedScore(mg + other.mg, eg + other.eg); }
    inline constexpr MixedScore operator-(const MixedScore& other) const { return MixedScore(mg - other.mg, eg - other.eg); }
    inline constexpr MixedScore operator*(const MixedScore& other) const { return MixedScore(mg * other.mg, eg * other.eg); }
    inline constexpr MixedScore operator/(const MixedScore& other) const { return MixedScore(mg / other.mg, eg / other.eg); }

    inline constexpr MixedScore operator+(int other) const { return MixedScore(mg + other, eg + other); }
    inline constexpr MixedScore operator-(int other) const { return MixedScore(mg - other, eg - other); }
    inline constexpr MixedScore operator*(int other) const { return MixedScore(mg * other, eg * other); }
    inline constexpr MixedScore operator/(int other) const { return MixedScore(mg / other, eg / other); }
};


constexpr MixedScore   PawnValue(  125,   200);
constexpr MixedScore KnightValue(  750,   850);
constexpr MixedScore BishopValue(  800,   900);
constexpr MixedScore   RookValue( 1200,  1400);
constexpr MixedScore  QueenValue( 2500,  2600);
constexpr MixedScore   KingValue(10000, 10000); // Needed for SEE and MVV-LVA

constexpr MixedScore piece_value[] = {
    PawnValue,
    KnightValue,
    BishopValue,
    RookValue,
    QueenValue,
    KingValue,
    MixedScore(0, 0), // Empty
    MixedScore(0, 0)  // PIECE_NONE
}; 

constexpr Score piece_value_mg[] = {
    PawnValue.middlegame(),
    KnightValue.middlegame(),
    BishopValue.middlegame(),
    RookValue.middlegame(),
    QueenValue.middlegame(),
    KingValue.middlegame(),
    0, // Empty
    0  // PIECE_NONE
};

constexpr Score piece_value_eg[] = {
    PawnValue.endgame(),
    KnightValue.endgame(),
    BishopValue.endgame(),
    RookValue.endgame(),
    QueenValue.endgame(),
    KingValue.endgame(),
    0, // Empty
    0  // PIECE_NONE
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


constexpr Piece get_piece(PieceType pt, Turn turn) { return static_cast<Piece>(pt + (turn << 3)); }
constexpr PieceType get_piece_type(Piece pc) { return static_cast<PieceType>(pc & 0b111); }
constexpr Turn get_turn(Piece pc) { return static_cast<Turn>((pc & 0b1000) >> 3); }


constexpr bool is_mate(const Score& score)
{
    Score abs_score = (score >= 0 ? score : -score);
    return abs_score >= SCORE_MATE_FOUND && abs_score <= SCORE_MATE;
}


constexpr Score score_to_tt(const Score& score, int ply)
{
    if (!is_mate(score))
        return score;

    return (score > 0) ? (score + ply) : (score - ply);
}


constexpr Score score_from_tt(const Score& score, int ply, int half_move)
{
    if (score >= SCORE_MATE_FOUND)
        return (SCORE_MATE - score > 99 - half_move) ? (SCORE_MATE_FOUND - 1)
                                                     : (score - ply);
    else if (score <= -SCORE_MATE_FOUND)
        return (SCORE_MATE + score > 99 - half_move) ? -(SCORE_MATE_FOUND - 1)
                                                     : (score + ply);
    else
        return score;
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
    return make_square(7 - rank(square), file(square));
}


inline constexpr Square horizontal_mirror(Square square)
{
    return make_square(rank(square), 7 - file(square));
}

template<Turn TURN>
inline int rank(Square s) { return TURN == WHITE ? rank(s) : 7 - rank(s); }

constexpr int rank(Square s, Turn turn) { return turn == WHITE ? rank(s) : 7 - rank(s); }

std::string get_square(Square square);


inline constexpr Square get_rook_square(CastleFile f, Turn turn)
{
    return (turn == WHITE ? SQUARE_A1 : SQUARE_A8) + static_cast<int>(f) - 1;
}


// Pseudo random number generator based on SplitMix64
class PseudoRandom
{
    uint64_t m_state;

public:
    inline PseudoRandom(uint64_t seed) : m_state(seed) {}

    inline static uint64_t get(uint64_t seed)
    {
        uint64_t z = seed + 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    inline uint64_t next()
    {
        uint64_t z = (m_state += 0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    inline uint64_t next(uint64_t n) { return next() % n; }
};


// Debug tools
namespace Debug
{
    enum Slots
    {
        NUM_DEBUG_SLOTS
    };


    constexpr std::array<const char*, NUM_DEBUG_SLOTS> Names
    {
    };


    constexpr bool Enabled = NUM_DEBUG_SLOTS > 0;


    class Entry
    {
        std::atomic_int64_t m_hits;
        std::atomic_int64_t m_sums;
        std::atomic_int64_t m_sums_squared;

    public:
        inline Entry() : m_hits(0), m_sums(0), m_sums_squared(0) {}
        inline void clear()
        {
            m_hits.store(0);
            m_sums.store(0);
            m_sums_squared.store(0);
        }
        inline void hit(int64_t value = 0)
        {
            m_hits.fetch_add(1, std::memory_order_relaxed);
            m_sums.fetch_add(value, std::memory_order_relaxed);
            m_sums_squared.fetch_add(value * value, std::memory_order_relaxed);
        }
        inline int64_t get_hits() const { return m_hits.load(); }
        inline int64_t get_sums() const { return m_sums.load(); }
        inline double get_avg() const { return double(m_sums.load()) / m_hits.load(); }
        inline double get_std() const
        {
            double avg = get_avg();
            return sqrt(double(m_sums_squared.load()) / m_hits.load() - avg * avg);
        }
    };


    extern std::array<Entry, NUM_DEBUG_SLOTS> debug_slots;


    inline void clear_debug_data()
    {
        for (auto& slot : debug_slots)
            slot.clear();
    }


    inline void print_debug_data()
    {
        std::cout << "Debug info:" << std::endl;
        for (std::size_t i = 0; i < NUM_DEBUG_SLOTS; i++)
            std::cout << " Slot: " << i
                      << " (" << Names[i] << ")"
                      << " Hits: " << debug_slots[i].get_hits()
                      << " Sums: " << debug_slots[i].get_sums()
                      << " Avg: "  << debug_slots[i].get_avg()
                      << " Std: "  << debug_slots[i].get_std()
                      << std::endl;
    }


    template<Slots Slot>
    void debug_hit(int64_t value = 0)
    {
        static_assert(Slot < NUM_DEBUG_SLOTS && Slot >= 0, "Invalid debug slot");
        debug_slots[Slot].hit(value);
    }
}
