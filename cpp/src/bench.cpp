#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
constexpr std::size_t n = 1u << 20;
constexpr int warmup = 8;
constexpr int iterations = 64;
constexpr std::array<std::uint16_t, 8> half_values{
    0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400};
volatile float sink = 0.0f;
volatile std::uint64_t sink_u64 = 0;

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

    std::vector<std::uint8_t> bytes_a(n), bytes_b(n);
    for (std::size_t i = 0; i < n; ++i) {
        bytes_a[i] = static_cast<std::uint8_t>((i * 17 + 3) & 255);
        bytes_b[i] = static_cast<std::uint8_t>((i * 29 + 11) & 255);
    }
    if (simd_lab::sad_u8_scalar(bytes_a, bytes_b) != simd_lab::sad_u8_best(bytes_a, bytes_b)) {
        std::cerr << "u8 SAD validation failed\n";
        return 1;
    }

    t = measure([&] { sink_u64 = simd_lab::sad_u8_scalar(bytes_a, bytes_b); });
    report("sad-u8/scalar-autovec", t, n * 2);

    t = measure([&] { sink_u64 = simd_lab::sad_u8_best(bytes_a, bytes_b); });
    report("sad-u8/best-dispatch", t, n * 2);

    std::vector<std::uint16_t> c(n), lo(n, 0x3800), hi(n, 0x4000), half_dst(n);
    for (std::size_t i = 0; i < n; ++i) c[i] = half_values[i & 7];

    if (simd_lab::clamp_f16c(half_dst.data(), c.data(), lo.data(), hi.data(), n)) {
        for (std::size_t i = 0; i < n; ++i) {
            if (half_dst[i] != std::clamp(c[i], std::uint16_t{0x3800}, std::uint16_t{0x4000})) {
                std::cerr << "F16C clamp validation failed at " << i << '\n';
                return 1;
            }
        }
        t = measure([&] {
            if (!simd_lab::clamp_f16c(half_dst.data(), c.data(), lo.data(), hi.data(), n)) {
                std::abort();
            }
        });
        report("clamp-f16/f16c-f32", t, n * 8);
    } else {
        std::cout << "clamp-f16/f16c-f32         skipped (AVX+F16C unavailable)\n";
    }

    std::cout << "N=" << n << " warmup=" << warmup
              << " iterations=" << iterations << '\n';
}
