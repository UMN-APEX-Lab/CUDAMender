#include "lu_factorization.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <nvtx3/nvToolsExt.h>

struct TestCase {
  std::string name;
  int m;
  std::vector<double> A;
  bool pivot_on;
  std::vector<double> expected_LU;
  std::vector<int> expected_Ipiv;
  int expected_info;
};

bool run_single_test(const TestCase& test_case) {
  std::cout << "\n========================================\n";
  std::cout << "Running test: " << test_case.name << "\n";
  std::cout << "Matrix size: " << test_case.m << "x" << test_case.m << "\n";
  std::cout << "Pivoting: " << (test_case.pivot_on ? "ON" : "OFF") << "\n";
  std::cout << "========================================\n";

  std::vector<double> LU(test_case.m * test_case.m, 0);
  std::vector<int> Ipiv(test_case.m, 0);
  int info = 0;

  lu_factorization(test_case.m, test_case.A, LU, Ipiv, info, test_case.pivot_on);

  // Check info code
  if (info != test_case.expected_info) {
    std::cerr << "Test FAILED: Info mismatch (expected " << test_case.expected_info 
              << ", got " << info << ")\n";
    return false;
  }

  if (info != 0) {
    std::cerr << "LU factorization failed with info = " << info << std::endl;
    return false;
  }

  std::cout << "LU matrix (column-major format):\n";
  for (int i = 0; i < test_case.m; ++i) {
    for (int j = 0; j < test_case.m; ++j) {
      printf("%f ", LU[i + j * test_case.m]);
    }
    printf("\n");
  }

  std::cout << "Pivoting sequence:\n";
  for (int i = 0; i < test_case.m; ++i) {
    printf("%d ", Ipiv[i]);
  }
  printf("\n");

  const double tolerance = 1e-5;
  bool passed = true;

  // Verify pivot indices
  for (int i = 0; i < test_case.m; ++i) {
    if (Ipiv[i] != test_case.expected_Ipiv[i]) {
      std::cerr << "Test FAILED: Pivot mismatch at index " << i 
                << " (expected " << test_case.expected_Ipiv[i] << ", got " << Ipiv[i] << ")\n";
      passed = false;
      break;
    }
  }

  // Verify LU matrix
  if (passed) {
    for (int i = 0; i < test_case.m * test_case.m; ++i) {
      if (std::abs(LU[i] - test_case.expected_LU[i]) > tolerance) {
        std::cerr << "Test FAILED: LU matrix mismatch at index " << i 
                  << " (expected " << test_case.expected_LU[i] << ", got " << LU[i] << ")\n";
        passed = false;
        break;
      }
    }
  }

  if (passed) {
    std::cout << "✓ Test PASSED: " << test_case.name << "\n";
  } else {
    std::cerr << "✗ Test FAILED: " << test_case.name << "\n";
  }

  return passed;
}

void run_benchmark() {
  const int m = 1024;
  const int n = m * m;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> A(n);
  for (int i = 0; i < n; ++i) {
    A[i] = dist(rng);
  }
  for (int i = 0; i < m; ++i) {
    A[i + i * m] += static_cast<double>(m);
  }

  std::vector<double> LU(n);
  std::vector<int> Ipiv(m);
  int info = 0;

  const int warmup = 3;
  const int timed = 100;

  for (int i = 0; i < warmup; ++i) {
    lu_factorization(m, A, LU, Ipiv, info, true);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; ++i) {
    lu_factorization(m, A, LU, Ipiv, info, true);
  }
  nvtxRangePop();
}

int main(int argc, char* argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  std::vector<TestCase> test_cases;

  // Test Case 1: 3x3 matrix with pivoting ON
  test_cases.push_back({
    "3x3 Matrix - Pivoting ON",
    3,  // m
    {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0},  // A (column-major)
    true,  // pivot_on
    // Expected LU matrix (column-major order)
    {
      7.0, 0.142857, 0.571429,      // column 0
      8.0, 0.857143, 0.500000,      // column 1
      10.0, 1.571429, -0.500000     // column 2
    },
    {3, 3, 3},  // expected_Ipiv
    0  // expected_info (success)
  });

  // Test Case 2: 3x3 matrix with pivoting OFF
  test_cases.push_back({
    "3x3 Matrix - Pivoting OFF",
    3,  // m
    {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0},  // A (column-major)
    false,  // pivot_on
    // Expected LU matrix without pivoting (column-major order)
    // Output matrix:
    // 1.0   2.0   3.0
    // 4.0  -3.0  -6.0
    // 7.0   2.0   1.0
    {
      1.0, 4.0, 7.0,                // column 0
      2.0, -3.0, 2.0,               // column 1
      3.0, -6.0, 1.0                // column 2
    },
    {0, 0, 0},  // expected_Ipiv (no pivoting)
    0  // expected_info (success)
  });

  // Test Case 3: 2x2 matrix with pivoting ON
  test_cases.push_back({
    "2x2 Matrix - Pivoting ON",
    2,  // m
    {4.0, 3.0, 6.0, 5.0},  // A (column-major): [[4,6],[3,5]]
    true,  // pivot_on
    // Expected LU matrix
    {
      4.0, 0.75,    // column 0
      6.0, 0.5      // column 1
    },
    {1, 2},  // expected_Ipiv
    0  // expected_info (success)
  });

  // Test Case 4: 2x2 matrix with pivoting OFF
  test_cases.push_back({
    "2x2 Matrix - Pivoting OFF",
    2,  // m
    {4.0, 3.0, 6.0, 5.0},  // A (column-major): [[4,6],[3,5]]
    false,  // pivot_on
    // Expected LU matrix without pivoting
    {
      4.0, 0.75,    // column 0
      6.0, 0.5      // column 1
    },
    {0, 0},  // expected_Ipiv (no pivoting)
    0  // expected_info (success)
  });

  // Test Case 5: 4x4 matrix with pivoting ON
  test_cases.push_back({
    "4x4 Matrix - Pivoting ON",
    4,  // m
    {
      2.0, 1.0, 1.0, 0.0,   // column 0
      4.0, 3.0, 3.0, 1.0,   // column 1
      8.0, 7.0, 9.0, 5.0,   // column 2
      6.0, 7.0, 9.0, 8.0    // column 3
    },
    true,  // pivot_on
    // Expected LU matrix with pivoting (from actual output)
    // Output matrix:
    // 2.0  4.0  8.0  6.0
    // 0.5  1.0  3.0  4.0
    // 0.5  1.0  2.0  2.0
    // 0.0  1.0  1.0  2.0
    {
      2.0, 0.5, 0.5, 0.0,     // column 0
      4.0, 1.0, 1.0, 1.0,     // column 1
      8.0, 3.0, 2.0, 1.0,     // column 2
      6.0, 4.0, 2.0, 2.0      // column 3
    },
    {1, 2, 3, 4},  // expected_Ipiv
    0  // expected_info (success)
  });

  // Test Case 6: 4x4 matrix with pivoting OFF
  test_cases.push_back({
    "4x4 Matrix - Pivoting OFF",
    4,  // m
    {
      2.0, 1.0, 1.0, 0.0,   // column 0
      4.0, 3.0, 3.0, 1.0,   // column 1
      8.0, 7.0, 9.0, 5.0,   // column 2
      6.0, 7.0, 9.0, 8.0    // column 3
    },
    false,  // pivot_on
    // Expected LU matrix without pivoting (from actual output)
    // Output matrix:
    // 2.0  4.0  8.0  6.0
    // 0.5  1.0  3.0  4.0
    // 0.5  1.0  2.0  2.0
    // 0.0  1.0  1.0  2.0
    {
      2.0, 0.5, 0.5, 0.0,     // column 0
      4.0, 1.0, 1.0, 1.0,     // column 1
      8.0, 3.0, 2.0, 1.0,     // column 2
      6.0, 4.0, 2.0, 2.0      // column 3
    },
    {0, 0, 0, 0},  // expected_Ipiv (no pivoting)
    0  // expected_info (success)
  });

  // Run all test cases
  for (const auto& test_case : test_cases) {
    if (!run_single_test(test_case)) {
      // Exit immediately with non-zero status on first failure
      std::cerr << "\n========================================\n";
      std::cerr << "TEST FAILED - EXITING\n";
      std::cerr << "========================================\n";
      std::exit(1);
    }
  }

  // Print summary - only reached if all tests pass
  std::cout << "\n========================================\n";
  std::cout << "TEST SUMMARY\n";
  std::cout << "========================================\n";
  std::cout << "Total tests: " << test_cases.size() << "\n";
  std::cout << "All tests PASSED!\n";
  std::cout << "========================================\n";

  return 0;
}