#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "ashihmin_d_mult_matr_crs/common/include/common.hpp"
#include "ashihmin_d_mult_matr_crs/omp/include/ops_omp.hpp"
#include "ashihmin_d_mult_matr_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_mult_matr_crs {

CRSMatrix DenseToCRS(const DenseMatrix &dense) {
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

bool CompareCRS(const CRSMatrix &A, const CRSMatrix &B) {
  if (A.rows != B.rows || A.cols != B.cols) {
    return false;
  }
  if (A.row_ptr != B.row_ptr) {
    return false;
  }
  if (A.col_index != B.col_index) {
    return false;
  }
  if (A.values.size() != B.values.size()) {
    return false;
  }

  for (size_t i = 0; i < A.values.size(); ++i) {
    if (std::abs(A.values[i] - B.values[i]) > 1e-8) {
      return false;
    }
  }
  return true;
}

using TestType = std::tuple<std::string, DenseMatrix, DenseMatrix, DenseMatrix>;

class AshihminDMultMatrCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &param) {
    return std::get<0>(param);
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    const auto &A = std::get<1>(params);
    const auto &B = std::get<2>(params);
    const auto &C = std::get<3>(params);

    input_data_ = std::make_pair(DenseToCRS(A), DenseToCRS(B));
    expected_output_ = DenseToCRS(C);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return CompareCRS(output_data, expected_output_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  CRSMatrix expected_output_;
};

TEST_P(AshihminDMultMatrCrsFuncTests, SpGemmCrs) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParams = {
    std::make_tuple("Identity_2x2", DenseMatrix{{1, 0}, {0, 1}}, DenseMatrix{{5, 6}, {7, 8}},
                    DenseMatrix{{5, 6}, {7, 8}}),

    std::make_tuple("ZeroMatrix", DenseMatrix{{0, 0}, {0, 0}}, DenseMatrix{{1, 2}, {3, 4}},
                    DenseMatrix{{0, 0}, {0, 0}}),

    std::make_tuple("Simple_2x2", DenseMatrix{{1, 2}, {3, 4}}, DenseMatrix{{5, 6}, {7, 8}},
                    DenseMatrix{{19, 22}, {43, 50}}),

    std::make_tuple("Rectangular", DenseMatrix{{1, 0, 2}, {0, 3, 0}}, DenseMatrix{{0, 1}, {4, 0}, {5, 6}},
                    DenseMatrix{{10, 13}, {12, 0}}),

    std::make_tuple("Diagonal", DenseMatrix{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}},
                    DenseMatrix{{4, 0, 0}, {0, 5, 0}, {0, 0, 6}}, DenseMatrix{{4, 0, 0}, {0, 10, 0}, {0, 0, 18}}),

    std::make_tuple("Single", DenseMatrix{{7}}, DenseMatrix{{8}}, DenseMatrix{{56}})};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AshihminDMultMatrCrsSEQ, AshihminDMultMatrCrsOMP>(
    kTestParams, PPC_SETTINGS_ashihmin_d_mult_matr_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(AshihminSparseCRSTests, AshihminDMultMatrCrsFuncTests, kGtestValues,
                         AshihminDMultMatrCrsFuncTests::PrintFuncTestName<AshihminDMultMatrCrsFuncTests>);

}  // namespace ashihmin_d_mult_matr_crs
