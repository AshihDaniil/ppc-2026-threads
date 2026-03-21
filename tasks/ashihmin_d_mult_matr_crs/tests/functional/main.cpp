#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "ashihmin_d_mult_matr_crs/common/include/common.hpp"
#include "ashihmin_d_mult_matr_crs/omp/include/ops_omp.hpp"
#include "ashihmin_d_mult_matr_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_mult_matr_crs {

inline CRSMatrix DenseToCRS(const DenseMatrix &dense) {
  CRSMatrix res;
  res.rows = static_cast<int>(dense.size());
  res.cols = dense.empty() ? 0 : static_cast<int>(dense[0].size());
  res.row_ptr.resize(res.rows + 1, 0);

  for (int i = 0; i < res.rows; ++i) {
    for (int j = 0; j < res.cols; ++j) {
      if (std::abs(dense[i][j]) > 1e-12) {
        res.values.push_back(dense[i][j]);
        res.col_index.push_back(j);
      }
    }
    res.row_ptr[i + 1] = static_cast<int>(res.values.size());
  }

  return res;
}

using InType = std::pair<CRSMatrix, CRSMatrix>;
using OutType = CRSMatrix;

class AshihminDMultMatrCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, InType> {
 public:
  static std::string PrintTestParam(const InType &) {
    return "SparseCRS";
  }

 protected:
  void SetUp() override {
    auto param = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = param;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const auto &A = input_data_.first;
    const auto &B = input_data_.second;

    std::vector<double> expected_vals;
    std::vector<int> expected_cols;
    std::vector<int> expected_ptr(A.rows + 1, 0);

    for (int i = 0; i < A.rows; ++i) {
      std::map<int, double> acc;

      for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
        int col_a = A.col_index[j];
        double val_a = A.values[j];

        for (int k = B.row_ptr[col_a]; k < B.row_ptr[col_a + 1]; ++k) {
          int col_b = B.col_index[k];
          double val_b = B.values[k];
          acc[col_b] += val_a * val_b;
        }
      }

      for (const auto &[col, val] : acc) {
        if (std::abs(val) > 1e-10) {
          expected_cols.push_back(col);
          expected_vals.push_back(val);
        }
      }

      expected_ptr[i + 1] = static_cast<int>(expected_vals.size());
    }

    return output_data.row_ptr == expected_ptr && output_data.col_index == expected_cols &&
           output_data.values.size() == expected_vals.size();
  }

 private:
  InType input_data_;
};

TEST_P(AshihminDMultMatrCrsFuncTests, SpGemmCRS) {
  ExecuteTest(GetParam());
}

static const std::array<InType, 6> kTestInputs = {
    InType{DenseToCRS({{1, 0}, {0, 1}}), DenseToCRS({{5, 6}, {7, 8}})},

    InType{DenseToCRS({{0, 0}, {0, 0}}), DenseToCRS({{1, 2}, {3, 4}})},

    InType{DenseToCRS({{1, 2}, {3, 4}}), DenseToCRS({{5, 6}, {7, 8}})},

    InType{DenseToCRS({{1, 0, 2}, {0, 3, 0}}), DenseToCRS({{0, 1}, {4, 0}, {5, 6}})},

    InType{DenseToCRS({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}), DenseToCRS({{4, 0, 0}, {0, 5, 0}, {0, 0, 6}})},

    InType{DenseToCRS({{7}}), DenseToCRS({{8}})}};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<AshihminDMultMatrCrsSEQ, AshihminDMultMatrCrsOMP, InType>(
        kTestInputs, PPC_SETTINGS_ashihmin_d_mult_matr_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(AshihminSparseCRSTests, AshihminDMultMatrCrsFuncTests, kGtestValues,
                         AshihminDMultMatrCrsFuncTests::PrintFuncTestName<AshihminDMultMatrCrsFuncTests>);

}  // namespace ashihmin_d_mult_matr_crs
