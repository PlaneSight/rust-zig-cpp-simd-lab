#include "kernels.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
constexpr std::size_t n = 1u << 20;
constexpr int warmup = 8;
constexpr int iterations = 64;
volatile float sink = 0.0f;

template <class F>
auto measure(F&& fn) {
    for (int i = 0; i < warmup; ++i) fn();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) fn();
    return std::chrono::steady_clock::now() - start;
}

void report(std::string_view name, std::chrono::steady_clock::duration elapsed,
            std::size_t bytes_per_iter) {
    const double seconds = std::chrono::duration<double>(elapsed).count();
    const double ns_elem = seconds * 1e9 / static_cast<double>(iterations * n);
    const double gib_s = static_cast<double>(bytes_per_iter * iterations) / seconds /
                         static_cast<double>(1ull << 30);
    std::cout << std::left << std::setw(28) << name << std::right
              << std::fixed << std::setprecision(4) << std::setw(9) << ns_elem
              << " ns/elem  " << std::setprecision(2) << std::setw(8) << gib_s
              << " GiB/s\n";
}
}

int main() {
    std::vector<float> x(n), y(n), dst(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) * 0.001f;
        y[i] = 1.0f + static_cast<float>(i) * 0.0005f;
    }

    auto t = measure([&] { simd_lab::axpy_scalar(dst, x, y, 0.75f); });
    report("axpy/scalar-autovec", t, n * 12);

    t = measure([&] { sink = simd_lab::squared_error_scalar(x, y); });
    report("sqerr/scalar-autovec", t, n * 8);

    t = measure([&] { sink = simd_lab::squared_error_best(x, y); });
    report("sqerr/best-dispatch", t, n * 8);

    std::cout << "N=" << n << " warmup=" << warmup
              << " iterations=" << iterations << '\n';
}
