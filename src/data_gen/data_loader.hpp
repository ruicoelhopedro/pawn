#pragma once

#include "Types.hpp"
#include "Position.hpp"
#include <string>
#include <vector>



extern "C"
{
    using Size = unsigned long long;

    void init_pawn();
    Size get_num_games(const char* file_name);
    void get_indices(const char* file_name, Size* indices);
    Size get_num_positions(const Size* indices, const Size* selection, Size n_selected);
    Size get_dense_psq_data(const char* file_name, const Size* selection, Size n_selected, char* X, short* evals, char* results);
    Size get_sparse_psq_data(const char* file_name, const Size* selection, Size n_selected, Size* idx, unsigned int* cols, char* colors, short* evals, char* results);
}