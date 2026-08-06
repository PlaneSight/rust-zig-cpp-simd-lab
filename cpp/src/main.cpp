#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    constexpr std::size_t n = 1u << 20;
    std::vector<float> x(n), y(n), dst(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) * 0.001f;
        y[i] = 1.0f + static_cast<float>(i) * 0.0005f;
    }

    simd_lab::axpy_scalar(dst, x, y, 0.75f);
    const float scalar = simd_lab::squared_error_scalar(x, y);
    const float best = simd_lab::squared_error_best(x, y);

    const double checksum = std::accumulate(dst.begin(), dst.end(), 0.0);
    const float tolerance = std::max(std::abs(scalar), 1.0f) * 1e-5f;

    if (std::abs(scalar - best) > tolerance) {
        std::cerr << "SIMD result mismatch: " << scalar << " vs " << best << '\n';
        return 1;
    }

    std::cout << "C++23 SIMD lab smoke test\n"
              << "AXPY checksum: " << checksum << '\n'
              << "Squared error scalar: " << scalar << '\n'
              << "Squared error best:   " << best << '\n';
}
