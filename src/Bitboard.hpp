#pragma once

#include "Types.hpp"
#include <inttypes.h>
#include <iostream>
#include <vector>

class Bitboard
{
	uint64_t m_data;

	// Index array for fast bitscan operations
	static constexpr int index64[64] = {
		 0, 47,  1, 56, 48, 27,  2, 60,
		57, 49, 41, 37, 28, 16,  3, 61,
		54, 58, 35, 52, 50, 42, 21, 44,
		38, 32, 29, 23, 17, 11,  4, 62,
		46, 55, 26, 59, 40, 36, 15, 53,
		34, 51, 20, 43, 31, 22, 10, 45,
		25, 39, 14, 33, 19, 30,  9, 24,
		13, 18,  8, 12,  7,  6,  5, 63 };

public:
	// IO operators
	friend std::ostream& operator<<(std::ostream& out, const Bitboard& bb);


	// Constructors
	constexpr Bitboard() : m_data(0) {}
	constexpr Bitboard(const Bitboard&) = default;
	constexpr Bitboard(uint64_t data) : m_data(data) {}


	// From single-bit
	static constexpr Bitboard from_single_bit(int index) { return Bitboard(uint64(1) << index); }
	static constexpr Bitboard from_square(Square square) { return from_single_bit(square); }


	// Bitwise operators
	constexpr operator bool() const
	{
		return m_data;
	}
	constexpr Bitboard& operator&=(Bitboard other) noexcept
	{
		this->m_data &= other.m_data;
		return *this;
	}
	constexpr Bitboard& operator|=(Bitboard other) noexcept
	{
		this->m_data |= other.m_data;
		return *this;
	}
	constexpr Bitboard& operator^=(Bitboard other) noexcept
	{
		this->m_data ^= other.m_data;
		return *this;
	}
	constexpr Bitboard& operator<<=(int i) noexcept
	{
		this->m_data <<= i;
		return *this;
	}
	constexpr Bitboard& operator>>=(int i) noexcept
	{
		this->m_data >>= i;
		return *this;
	}

	constexpr Bitboard operator~() const noexcept
	{
		return Bitboard(~m_data);
	}
	constexpr Bitboard operator&(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result &= other;
		return result;
	}
	constexpr Bitboard operator|(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result |= other;
		return result;
	}
	constexpr Bitboard operator^(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result ^= other;
		return result;
	}
	constexpr Bitboard operator<<(int i) const noexcept
	{
		Bitboard result = *this;
		result <<= i;
		return result;
	}
	constexpr Bitboard operator>>(int i) const noexcept
	{
		Bitboard result = *this;
		result >>= i;
		return result;
	}


	// Arithmetic operators
	constexpr Bitboard& operator+=(Bitboard other) noexcept
	{
		m_data += other.m_data;
		return *this;
	}
	constexpr Bitboard& operator-=(Bitboard other) noexcept
	{
		m_data -= other.m_data;
		return *this;
	}
	constexpr Bitboard& operator*=(Bitboard other) noexcept
	{
		m_data *= other.m_data;
		return *this;
	}
	constexpr Bitboard& operator/=(Bitboard other) noexcept
	{
		m_data /= other.m_data;
		return *this;
	}
	constexpr Bitboard operator+(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result += other;
		return result;
	}
	constexpr Bitboard operator-(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result -= other;
		return result;
	}
	constexpr Bitboard operator*(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result *= other;
		return result;
	}
	constexpr Bitboard operator/(Bitboard other) const noexcept
	{
		Bitboard result = *this;
		result /= other;
		return result;
	}
	constexpr bool operator==(Bitboard other) const noexcept
	{
		return m_data == other.m_data;
	}


	// Single-bit functions
	constexpr uint64_t single_bit(int i) const noexcept
	{
		return uint64(1) << i;
	}
	constexpr bool test(int i) const noexcept
	{
		return (m_data & single_bit(i));
	}
	constexpr void set(int i) noexcept
	{
		m_data |= single_bit(i);
	}
	constexpr void toggle(int i) noexcept
	{
		m_data ^= single_bit(i);
	}
	constexpr void reset(int i) noexcept
	{
		m_data &= ~single_bit(i);
	}


	// General tests
	constexpr bool is_zero() const noexcept
	{
		return m_data == 0;
	}
	constexpr bool is_full() const noexcept
	{
		return m_data == (~0);
	}
	constexpr int count() const noexcept
	{
#if defined(__GNUC__)
		return __builtin_popcountll(m_data);
#else
		int count = 0;
		Bitboard x = *this;
		while (x.m_data)
		{
			count++;
			x.reset_lsb();
		}
		return count;
#endif
	}
	constexpr uint64_t to_uint64() const noexcept
	{
		return m_data;
	}


	// Bit-scan
	constexpr int bitscan_forward() const noexcept
	{
#if defined(__GNUC__)
		return __builtin_ctzll(m_data);
#else
		const uint64_t magic = 0x03f79d71b4cb0a89;
		return index64[((m_data ^ (m_data - 1)) * magic) >> 58];
#endif
	}
	constexpr int bitscan_reverse() const noexcept
	{
		const uint64_t magic = 0x03f79d71b4cb0a89;
		auto bb = m_data;
		bb |= bb >> 1;
		bb |= bb >> 2;
		bb |= bb >> 4;
		bb |= bb >> 8;
		bb |= bb >> 16;
		bb |= bb >> 32;
		return index64[(bb * magic) >> 58];
	}
	constexpr int bitscan_forward_reset() noexcept
	{
		int index = bitscan_forward();
		reset_lsb();
		return index;
	}
	constexpr void reset_lsb() noexcept
	{
		m_data &= m_data - 1;
	}
	constexpr Bitboard below_lsb() const noexcept
	{
		return Bitboard(~m_data & (m_data - 1));
	}
	constexpr Bitboard below_lsb_including() const noexcept
	{
		return Bitboard(m_data ^ (m_data - 1));
	}
	constexpr bool more_than_one() const noexcept
	{
		Bitboard bb = *this;
		bb.reset_lsb();
		return bb;
	}


	// Shifting
	template<Direction DIR>
	constexpr Bitboard shift() const noexcept
	{
		if (DIR > 0)
			return (*this) << DIR;
		else
			return (*this) >> -DIR;
	}
	constexpr Bitboard shift(Direction dir) const noexcept
	{
		if (dir > 0)
			return (*this) << dir;
		else
			return (*this) >> -dir;
	}


	// Fill
	template<Direction DIR>
	constexpr Bitboard fill() const
	{
		Bitboard result = (*this);
		result |= result.shift<1 * DIR>();
		result |= result.shift<2 * DIR>();
		result |= result.shift<4 * DIR>();
		return result;
	}
	template<Direction DIR>
	constexpr Bitboard fill_excluded() const
	{
		Bitboard result = (*this).shift<DIR>();
		result |= result.shift<1 * DIR>();
		result |= result.shift<2 * DIR>();
		result |= result.shift<4 * DIR>();
		return result;
	}


	// Flip
	constexpr Bitboard vertical_flip() const
	{
		constexpr Bitboard k1(0x00FF00FF00FF00FF);
		constexpr Bitboard k2(0x0000FFFF0000FFFF);
		Bitboard result = *this;
		result = ((result >> 8)  & k1) | ((result & k1) << 8);
		result = ((result >> 16) & k2) | ((result & k2) << 16);
		result = ( result >> 32)       | ( result       << 32);
		return result;
	}
	constexpr Bitboard horizontal_flip() const
	{
		constexpr Bitboard k1(0x5555555555555555);
		constexpr Bitboard k2(0x3333333333333333);
		constexpr Bitboard k4(0x0f0f0f0f0f0f0f0f);
		Bitboard result = *this;
		result = ((result >> 1) & k1) + (Bitboard( 2) * (result & k1));
		result = ((result >> 2) & k2) + (Bitboard( 4) * (result & k2));
		result = ((result >> 4) & k4) + (Bitboard(16) * (result & k4));
		return result;
	}
};

// IO operators
std::ostream& operator<<(std::ostream& out, const Bitboard& bb);


namespace Bitboards
{
	constexpr Bitboard empty = Bitboard(0);
	constexpr Bitboard full = ~Bitboard(0);

	constexpr Bitboard a_file(0x101010101010101ULL);
	constexpr Bitboard b_file = a_file << 1;
	constexpr Bitboard c_file = a_file << 2;
	constexpr Bitboard d_file = a_file << 3;
	constexpr Bitboard e_file = a_file << 4;
	constexpr Bitboard f_file = a_file << 5;
	constexpr Bitboard g_file = a_file << 6;
	constexpr Bitboard h_file = a_file << 7;

	constexpr Bitboard rank_1(0xFFULL);
	constexpr Bitboard rank_2 = rank_1 << (8 * 1);
	constexpr Bitboard rank_3 = rank_1 << (8 * 2);
	constexpr Bitboard rank_4 = rank_1 << (8 * 3);
	constexpr Bitboard rank_5 = rank_1 << (8 * 4);
	constexpr Bitboard rank_6 = rank_1 << (8 * 5);
	constexpr Bitboard rank_7 = rank_1 << (8 * 6);
	constexpr Bitboard rank_8 = rank_1 << (8 * 7);

	constexpr Bitboard files[8] = { a_file, b_file, c_file, d_file,
								    e_file, f_file, g_file, h_file };

	constexpr Bitboard ranks[8] = { rank_1, rank_2, rank_3, rank_4,
								    rank_5, rank_6, rank_7, rank_8 };

	constexpr Bitboard square_color[NUM_COLORS] = { 0xAA55AA55AA55AA55ULL, ~0xAA55AA55AA55AA55ULL };

	class MagicBitboard
	{
		int m_bits;
		uint64_t m_magic;
		Bitboard m_blockmask;
		std::vector<Bitboard> m_moveboards;
		int get_index(Bitboard blockboard) const;

	public:
		MagicBitboard() = default;
		MagicBitboard(uint64_t magic, Bitboard blockmask, const std::vector<Bitboard>& blockboards, const std::vector<Bitboard>& moveboards);

		Bitboard get_moveboard(Bitboard occupancy) const;
	};



	extern Bitboard diagonals[NUM_SQUARES];
	extern Bitboard ranks_files[NUM_SQUARES];

	extern Bitboard pseudo_attacks[NUM_PIECE_TYPES][NUM_SQUARES];
	extern Bitboard pawn_attacks[NUM_COLORS][NUM_SQUARES];

	extern Bitboard castle_non_attacked_squares[NUM_COLORS][NUM_CASTLE_SIDES];
	extern Bitboard castle_non_occupied_squares[NUM_COLORS][NUM_CASTLE_SIDES];
	extern Square castle_target_square[NUM_COLORS][NUM_CASTLE_SIDES];

	extern MagicBitboard bishop_magics[NUM_SQUARES];
	extern MagicBitboard rook_magics[NUM_SQUARES];

	extern Bitboard between_squares[NUM_SQUARES][NUM_SQUARES];

	template <Piece PIECE_TYPE>
	Bitboard get_attacks(Square square, Bitboard occupancy)
	{
		//static_assert(PIECE_TYPE != PAWN, "Not available for pawns!");
		return pseudo_attacks[PIECE_TYPE][square];
	}

	template<>
	Bitboard get_attacks<BISHOP>(Square square, Bitboard occupancy);

	template<>
	Bitboard get_attacks<ROOK>(Square square, Bitboard occupancy);

	template<>
	Bitboard get_attacks<QUEEN>(Square square, Bitboard occupancy);

	template <Turn TURN>
	Bitboard get_attacks_pawns(Square square)
	{
		return pawn_attacks[TURN][square];
	}


	Bitboard between(Square s1, Square s2);

	void build_diagonals();
	void build_ranks_files();

	void build_pseudo_attacks(Square square);

	template <Turn TURN>
	Bitboard pseudo_attacks_pawns(Square square)
	{
		Bitboard result = empty;
		Color color = turn_to_color(TURN);
		int i = rank(square);
		int j = file(square);
		for (int delta : { -1, 1 })
			if (inside_board(i + color, j + delta))
				result.set(make_square(i + color, j + delta));
		return result;
	}

	Bitboard pseudo_attacks_knights(Square square);
	Bitboard pseudo_attacks_bishops(Square square);
	Bitboard pseudo_attacks_rooks(Square square);
	Bitboard pseudo_attacks_queens(Square square);
	Bitboard pseudo_attacks_kings(Square square);

	void build_castle_squares();

	void build_between_bbs();

	void init_bitboards();


	namespace magic_helpers
	{
		// Pre-computed magic numbers
		constexpr uint64_t rook_magics[64] = { 2558044863226450048,
											   2323875000446107649,
											   108095599501377664,
											   4647719763383029760,
											   720585148857319936,
											   36030446353531392,
											   288340361942859906,
											   4791831382207766656,
											   579275665283809408,
											   4613093668264354048,
											   36310551236714496,
											   3765290802666934272,
											   563087665397768,
											   578149670950404104,
											   3837348499231605248,
											   10414584347295746,
											   36029071966077650,
											   4521466695786496,
											   22520747192754176,
											   67554544837464192,
											   2252349703979136,
											   282574623342594,
											   720580338459412624,
											   739155487872188548,
											   900728997518778368,
											   35211215654918,
											   22870945666433169,
											   283437778866176,
											   9297474619768848,
											   162692613849809936,
											   28673202196497,
											   316942817951889,
											   1152991873896284296,
											   1225054002878615552,
											   145258817624490240,
											   4613392606536155144,
											   11260115768313985,
											   576742244476782622,
											   72356721934205256,
											   19140590776093057,
											   324269352780578816,
											   4538785341784066,
											   27303210181001282,
											   1407718506692641,
											   2252920800739344,
											   2533310827921410,
											   360587042759114760,
											   83883942188548097,
											   739787166036533760,
											   1157434175207243840,
											   35734429909504,
											   602738547235072,
											   1162491688443453952,
											   2027182817435452928,
											   576478366100751360,
											   281519075754752,
											   36383943846527009,
											   36099238780076162,
											   35188684370113,
											   144151473184591873,
											   4757490641026166850,
											   1971037868660753,
											   8804700782724,
											   101331309997326470 };
		constexpr uint64_t bishop_magics[64] = { 1149058404452416,
												 289081415381942273,
												 6756241263034384,
												 11338992834838564,
												 37160212215365633,
												 4901615158836068353,
												 4617335344705634340,
												 5190421711845343236,
												 4416434471936,
												 1155261961169273012,
												 4630338168049172608,
												 8111620697219944448,
												 144119599143714816,
												 15428630349030,
												 72059827437973512,
												 1729666008228177920,
												 4503737106319360,
												 218424659270963714,
												 5190399412760809488,
												 586030918793297921,
												 145241122913387872,
												 4612820860489826816,
												 2310910711054992392,
												 2315131684025541632,
												 9015999912214787,
												 5664684175851776,
												 2326118072533991680,
												 563568562946240,
												 4611968593990533137,
												 256997648920871185,
												 4613115383589767200,
												 5927583733590085890,
												 2306415031480487936,
												 164946673555996964,
												 2918898811362345025,
												 4592694445998144,
												 4611756663127671040,
												 1155178806307719296,
												 4685996116670480640,
												 2307005271387808784,
												 156800460152651804,
												 96929493093380,
												 725101530843318280,
												 873698611186108416,
												 18579556780737536,
												 4702059285985627152,
												 1154119508625982729,
												 725127936784531584,
												 4647997424332249089,
												 293930536557084673,
												 2305926865565517825,
												 580964395153358848,
												 36037627522320388,
												 92376290951168,
												 1161963896857184288,
												 325403223614620208,
												 6921470778185883680,
												 292133800064,
												 73118664377344,
												 1441151889487036930,
												 4611686293381858304,
												 1161939785012494593,
												 5188217148471902368,
												 567416728125696 };

		Bitboard blockmask_rook(Square square);
		Bitboard blockmask_bishop(Square square);
		Bitboard moveboard_rook(Square square, Bitboard blockboard);
		Bitboard moveboard_bishop(Square square, Bitboard blockboard);
		std::vector<Bitboard> gen_blockboards(Bitboard blockmask);
		uint64_t gen_magic(Bitboard blockmask, std::vector<Bitboard> blockboards);

		void gen_all_magics(bool compute = false);

		uint64_t random_uint64();
		uint64_t random_uint64_fewbits();
	}
}
