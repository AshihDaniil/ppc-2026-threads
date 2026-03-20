#include "ashihmin_d_mult_matr_crs/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

namespace ashihmin_d_mult_matr_crs {

AshihminDMultMatrCrsOMP::AshihminDMultMatrCrsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool AshihminDMultMatrCrsOMP::ValidationImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  if (matrix_a.cols != matrix_b.rows) {
    return false;
  }

  if (matrix_a.row_ptr.size() != static_cast<std::size_t>(matrix_a.rows) + 1 ||
      matrix_b.row_ptr.size() != static_cast<std::size_t>(matrix_b.rows) + 1) {
    return false;
  }

  return true;
}

bool AshihminDMultMatrCrsOMP::PreProcessingImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  auto &matrix_c = GetOutput();
  matrix_c.rows = matrix_a.rows;
  matrix_c.cols = matrix_b.cols;
  matrix_c.row_ptr.assign(matrix_a.rows + 1, 0);
  matrix_c.values.clear();
  matrix_c.col_index.clear();

  return true;
}

bool AshihminDMultMatrCrsOMP::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;
  auto &matrix_c = GetOutput();

  int rows_a = matrix_a.rows;

  std::vector<std::vector<int>> local_cols(rows_a);
  std::vector<std::vector<double>> local_vals(rows_a);

#pragma omp parallel for

  for (int i = 0; i < rows_a; ++i) {
    std::map<int, double> row_accumulator;

    int row_a_start = matrix_a.row_ptr[i];
    int row_a_end = matrix_a.row_ptr[i + 1];

    for (int j = row_a_start; j < row_a_end; ++j) {
      int col_a = matrix_a.col_index[j];
      double val_a = matrix_a.values[j];

      int row_b_start = matrix_b.row_ptr[col_a];
      int row_b_end = matrix_b.row_ptr[col_a + 1];

      for (int k = row_b_start; k < row_b_end; ++k) {
        int col_b = matrix_b.col_index[k];
        double val_b = matrix_b.values[k];

        row_accumulator[col_b] += val_a * val_b;
      }
    }

    for (const auto &[col, val] : row_accumulator) {
      if (std::abs(val) > 1e-15) {
        local_cols[i].push_back(col);
        local_vals[i].push_back(val);
      }
    }
  }

  for (int i = 0; i < rows_a; ++i) {
    matrix_c.col_index.insert(matrix_c.col_index.end(), local_cols[i].begin(), local_cols[i].end());
    matrix_c.values.insert(matrix_c.values.end(), local_vals[i].begin(), local_vals[i].end());
    matrix_c.row_ptr[i + 1] = static_cast<int>(matrix_c.values.size());
  }

  return true;
}

bool AshihminDMultMatrCrsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_mult_matr_crs
