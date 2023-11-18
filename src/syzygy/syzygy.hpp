#include "../Types.hpp"
#include "../Position.hpp"
#include "../Hash.hpp"
#include <string>

namespace Syzygy
{
    enum WDL
    {
        WDL_LOSS = -2,
        WDL_BLESSED_LOSS = -1,
        WDL_DRAW = 0,
        WDL_CURSED_WIN = 1,
        WDL_WIN = 2,
        WDL_NONE = 3
    };
    inline WDL operator-(const WDL& wdl) { return static_cast<WDL>(-static_cast<int>(wdl)); }

    struct RootMove
    {
        Score tb_score;
        int dtz;
        Move move;
    };

    class RootPos
    {
        RootMove m_moves[NUM_MAX_MOVES];
        Score m_score;
        int m_num_moves;
        int m_num_preserving_moves;

    public:
        RootPos();
        RootPos(Position& pos);

        bool in_tb() const;
        int num_preserving_moves() const;
        Move ordered_moves(int idx) const;
        Score move_score(Move move) const;

    };

    extern bool Loaded;
    extern RootPos Root;
    extern int Cardinality;

    void init();

    void load(std::string path);

    void clear();

    WDL probe_wdl(const Board& board);

    std::pair<WDL, int> probe_dtz(const Board& board);

    WDL fathom_to_wdl(unsigned int result);

    Score score_from_wdl(WDL wdl, Depth ply = 0);

    EntryType bound_from_wdl(WDL wdl);
}
