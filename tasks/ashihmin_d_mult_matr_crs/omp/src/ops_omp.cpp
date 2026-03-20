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
  return GetInput().first.cols == GetInput().second.rows;
}

bool AshihminDMultMatrCrsOMP::PreProcessingImpl() {
  auto &matrix_c = GetOutput();
  matrix_c.rows = GetInput().first.rows;
  matrix_c.cols = GetInput().second.cols;
  matrix_c.row_ptr.assign(matrix_c.rows + 1, 0);
  matrix_c.values.clear();
  matrix_c.col_index.clear();
  return true;
}

bool AshihminDMultMatrCrsOMP::RunImpl() {
  const auto &A = GetInput().first;
  const auto &B = GetInput().second;
  auto &C = GetOutput();
  int rows_a = A.rows;

  std::vector<std::vector<int>> local_cols(rows_a);
  std::vector<std::vector<double>> local_vals(rows_a);

#pragma omp parallel for
  for (int i = 0; i < rows_a; ++i) {
    std::map<int, double> row_acc;
    for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
      int col_a = A.col_index[j];
      double val_a = A.values[j];
      for (int k = B.row_ptr[col_a]; k < B.row_ptr[col_a + 1]; ++k) {
        row_acc[B.col_index[k]] += val_a * B.values[k];
      }
    }
    for (const auto &[col, val] : row_acc) {
      if (std::abs(val) > 1e-15) {
        local_cols[i].push_back(col);
        local_vals[i].push_back(val);
      }
    }
  }

  for (int i = 0; i < rows_a; ++i) {
    C.col_index.insert(C.col_index.end(), local_cols[i].begin(), local_cols[i].end());
    C.values.insert(C.values.end(), local_vals[i].begin(), local_vals[i].end());
    C.row_ptr[i + 1] = static_cast<int>(C.values.size());
  }
  return true;
}

bool AshihminDMultMatrCrsOMP::PostProcessingImpl() {
  return true;
}

}  // namespace ashihmin_d_mult_matr_crs
