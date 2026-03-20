#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include "ashihmin_d_mult_matr_crs/common/include/common.hpp"
#include "ashihmin_d_mult_matr_crs/omp/include/ops_omp.hpp"
#include "ashihmin_d_mult_matr_crs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace ashihmin_d_mult_matr_crs {

inline CRSMatrix LocalDenseToCRS(const DenseMatrix &dense) {
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

struct TaskData {
  std::string name;
  InType input;
  OutType expected;
  operator InType() const {
    return input;
  }
};

class AshihminDMultMatrCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TaskData> {
 public:
  static std::string PrintTestParam(const TaskData &param) {
    return param.name;
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params.input;
    expected_output_ = params.expected;
  }
  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.row_ptr != expected_output_.row_ptr) {
      return false;
    }
    if (output_data.col_index != expected_output_.col_index) {
      return false;
    }
    for (size_t i = 0; i < output_data.values.size(); ++i) {
      if (std::abs(output_data.values[i] - expected_output_.values[i]) > 1e-9) {
        return false;
      }
    }
    return true;
  }
  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  CRSMatrix expected_output_;
};

static const std::vector<TaskData> kTestParams = {
    {"Identity_2x2",
     {LocalDenseToCRS({{1, 0}, {0, 1}}), LocalDenseToCRS({{5, 6}, {7, 8}})},
     LocalDenseToCRS({{5, 6}, {7, 8}})},
    {"ZeroMatrix",
     {LocalDenseToCRS({{0, 0}, {0, 0}}), LocalDenseToCRS({{1, 2}, {3, 4}})},
     LocalDenseToCRS({{0, 0}, {0, 0}})},
    {"Simple_2x2",
     {LocalDenseToCRS({{1, 2}, {3, 4}}), LocalDenseToCRS({{5, 6}, {7, 8}})},
     LocalDenseToCRS({{19, 22}, {43, 50}})},
    {"Rectangular_2x3_3x2",
     {LocalDenseToCRS({{1, 0, 2}, {0, 3, 0}}), LocalDenseToCRS({{0, 1}, {4, 0}, {5, 6}})},
     LocalDenseToCRS({{10, 13}, {12, 0}})},
    {"SparseDiagonal",
     {LocalDenseToCRS({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}), LocalDenseToCRS({{4, 0, 0}, {0, 5, 0}, {0, 0, 6}})},
     LocalDenseToCRS({{4, 0, 0}, {0, 10, 0}, {0, 0, 18}})},
    {"SingleElement", {LocalDenseToCRS({{7}}), LocalDenseToCRS({{8}})}, LocalDenseToCRS({{56}})}};

TEST_P(AshihminDMultMatrCrsFuncTests, SpGemm) {
  ExecuteTest(GetParam());
}

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<AshihminDMultMatrCrsSEQ, AshihminDMultMatrCrsOMP>(
    kTestParams, PPC_SETTINGS_ashihmin_d_mult_matr_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
INSTANTIATE_TEST_SUITE_P(AshihminSparseCRSTests, AshihminDMultMatrCrsFuncTests, kGtestValues,
                         AshihminDMultMatrCrsFuncTests::PrintFuncTestName<AshihminDMultMatrCrsFuncTests>);

}  // namespace ashihmin_d_mult_matr_crs
