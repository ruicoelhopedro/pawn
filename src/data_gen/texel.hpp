#pragma once

#include <sstream>

namespace Texel
{
    using Accumulator = std::uint64_t;

    void score_eval_error(std::istringstream& stream);

    void score_texel(std::istringstream& stream);
}
