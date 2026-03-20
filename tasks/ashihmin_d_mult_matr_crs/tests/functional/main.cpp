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

// Локальный конвертер
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

class AshihminDMultMatrCrsFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, InType> {
 public:
  // Для InType (std::pair) вывод имени не важен, возвращаем заглушку
  static std::string PrintTestParam(const InType &) {
    return "SparseMatrixTask";
  }

 protected:
  void SetUp() override {
    auto params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = params;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Вычисляем эталон прямо здесь, используя входные данные
    // Это единственный способ, если нельзя менять SEQ и его конструктор
    const auto &A = input_data_.first;
    const auto &B = input_data_.second;

    // Простая последовательная проверка результата
    std::vector<double> expected_vals;
    std::vector<int> expected_cols;
    std::vector<int> expected_ptr(A.rows + 1, 0);

    for (int i = 0; i < A.rows; ++i) {
      std::map<int, double> row_acc;
      for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
        for (int k = B.row_ptr[A.col_index[j]]; k < B.row_ptr[A.col_index[j] + 1]; ++k) {
          row_acc[B.col_index[k]] += A.values[j] * B.values[k];
        }
      }
      for (const auto &[col, val] : row_acc) {
        if (std::abs(val) > 1e-10) {
          expected_vals.push_back(val);
          expected_cols.push_back(col);
        }
      }
      expected_ptr[i + 1] = static_cast<int>(expected_vals.size());
    }

    if (output_data.row_ptr != expected_ptr) {
      return false;
    }
    if (output_data.col_index != expected_cols) {
      return false;
    }
    for (size_t i = 0; i < output_data.values.size(); ++i) {
      if (std::abs(output_data.values[i] - expected_vals[i]) > 1e-8) {
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
};

// Передаем СТРОГО InType (std::pair<CRSMatrix, CRSMatrix>)
static const std::array<InType, 6> kTestInputs = {
    {{LocalDenseToCRS({{1, 0}, {0, 1}}), LocalDenseToCRS({{5, 6}, {7, 8}})},
     {LocalDenseToCRS({{0, 0}, {0, 0}}), LocalDenseToCRS({{1, 2}, {3, 4}})},
     {LocalDenseToCRS({{1, 2}, {3, 4}}), LocalDenseToCRS({{5, 6}, {7, 8}})},
     {LocalDenseToCRS({{1, 0, 2}, {0, 3, 0}}), LocalDenseToCRS({{0, 1}, {4, 0}, {5, 6}})},
     {LocalDenseToCRS({{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}), LocalDenseToCRS({{4, 0, 0}, {0, 5, 0}, {0, 0, 6}})},
     {LocalDenseToCRS({{7}}), LocalDenseToCRS({{8}})}}};

TEST_P(AshihminDMultMatrCrsFuncTests, SpGemm) {
  ExecuteTest(GetParam());
}

// ТРЕТИЙ ПАРАМЕТР ШАБЛОНА - InType. Это гарантирует вызов конструктора SEQ(const InType&)
const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<AshihminDMultMatrCrsSEQ, AshihminDMultMatrCrsOMP, InType>(
        kTestInputs, PPC_SETTINGS_ashihmin_d_mult_matr_crs));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
INSTANTIATE_TEST_SUITE_P(AshihminSparseCRSTests, AshihminDMultMatrCrsFuncTests, kGtestValues,
                         AshihminDMultMatrCrsFuncTests::PrintFuncTestName<AshihminDMultMatrCrsFuncTests>);

}  // namespace ashihmin_d_mult_matr_crs
