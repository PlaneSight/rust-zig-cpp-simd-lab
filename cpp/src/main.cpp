#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

class XorShift64 {
public:
    explicit XorShift64(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() noexcept {
        auto value = state_;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        state_ = value;
        return value;
    }

    float next_float() noexcept {
        constexpr float scale = 1.0F / static_cast<float>(1U << 24);
        const auto unit = static_cast<float>(next() >> 40) * scale;
        return std::fma(unit, 8.0F, -4.0F);
    }

private:
    std::uint64_t state_;
};

bool run_randomized_differential_tests() {
    constexpr std::array<std::uint16_t, 8> half_values{
        0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400};
    XorShift64 rng{0x8f3ca516d27b49e1ULL};

    for (std::size_t trial = 0; trial < 256; ++trial) {
        const std::size_t length = trial < 64 ? trial : rng.next() % 2049;
        std::vector<float> a(length), b(length);
        std::generate(a.begin(), a.end(), [&] { return rng.next_float(); });
        std::generate(b.begin(), b.end(), [&] { return rng.next_float(); });
        const auto reference = simd_lab::squared_error_scalar(a, b);
        const auto candidate = simd_lab::squared_error_best(a, b);
        const auto relative_error = std::abs(reference - candidate) /
                                    std::max(std::abs(reference), 1.0);
        if (relative_error > 1e-12) {
            std::cerr << "squared-error mismatch at length " << length << '\n';
            return false;
        }

        std::vector<std::uint8_t> bytes_a(length), bytes_b(length);
        std::generate(bytes_a.begin(), bytes_a.end(), [&] {
            return static_cast<std::uint8_t>(rng.next());
        });
        std::generate(bytes_b.begin(), bytes_b.end(), [&] {
            return static_cast<std::uint8_t>(rng.next());
        });
        if (simd_lab::sad_u8_scalar(bytes_a, bytes_b) !=
            simd_lab::sad_u8_best(bytes_a, bytes_b)) {
            std::cerr << "SAD mismatch at length " << length << '\n';
            return false;
        }

        std::vector<std::uint8_t> expected(length), sat_candidate(length);
        for (std::size_t i = 0; i < length; ++i) {
            const auto widened_sum = static_cast<unsigned>(bytes_a[i]) +
                                     static_cast<unsigned>(bytes_b[i]);
            expected[i] = static_cast<std::uint8_t>(
                std::min(widened_sum, 255U));
        }
        simd_lab::sat_add_u8_best(sat_candidate, bytes_a, bytes_b);
        if (sat_candidate != expected) {
            std::cerr << "saturating-add mismatch at length " << length << '\n';
            return false;
        }

        std::vector<std::uint16_t> c(length), lo(length, 0x3800),
            hi(length, 0x4000), dst(length, 0xdead);
        for (std::size_t i = 0; i < length; ++i) {
            c[i] = half_values[(rng.next() + i) & 7];
        }
        const auto before = dst;
        const bool ran = simd_lab::clamp_f16c(
            dst.data(), c.data(), lo.data(), hi.data(), length);
        if (length % 8 != 0) {
            if (ran || dst != before) {
                std::cerr << "F16C rejection mutated dst at length " << length << '\n';
                return false;
            }
        } else if (ran) {
            for (std::size_t i = 0; i < length; ++i) {
                if (dst[i] != std::clamp(c[i], std::uint16_t{0x3800},
                                        std::uint16_t{0x4000})) {
                    std::cerr << "F16C mismatch at length " << length
                              << ", index " << i << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

bool run_exhaustive_saturating_add_test() {
    constexpr std::size_t pair_count = 256 * 256;
    std::vector<std::uint8_t> a(pair_count), b(pair_count),
        expected(pair_count), candidate(pair_count);

    for (std::size_t x = 0; x < 256; ++x) {
        for (std::size_t y = 0; y < 256; ++y) {
            const auto index = x * 256 + y;
            a[index] = static_cast<std::uint8_t>(x);
            b[index] = static_cast<std::uint8_t>(y);
            expected[index] = static_cast<std::uint8_t>(
                std::min(x + y, std::size_t{255}));
        }
    }

    simd_lab::sat_add_u8_scalar(candidate, a, b);
    if (candidate != expected) {
        std::cerr << "scalar saturating-add exhaustive test failed\n";
        return false;
    }
    std::fill(candidate.begin(), candidate.end(), std::uint8_t{0});
    simd_lab::sat_add_u8_best(candidate, a, b);
    if (candidate != expected) {
        std::cerr << "dispatched saturating-add exhaustive test failed\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    constexpr std::size_t n = 1u << 20;
    std::vector<float> x(n), y(n), dst(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i) * 0.001F;
        y[i] = 1.0F + static_cast<float>(i) * 0.0005F;
    }

    simd_lab::axpy_scalar(dst, x, y, 0.75F);
    const auto scalar = simd_lab::squared_error_scalar(x, y);
    const auto best = simd_lab::squared_error_best(x, y);
    const auto checksum = std::accumulate(dst.begin(), dst.end(), 0.0);
    const auto relative_error = std::abs(scalar - best) /
                                std::max(std::abs(scalar), 1.0);

    if (relative_error > 1e-12 || !run_randomized_differential_tests() ||
        !run_exhaustive_saturating_add_test()) {
        return 1;
    }

    std::cout << "C++23 SIMD lab smoke test\n"
              << "Dispatch tier: " << simd_lab::dispatch_tier() << '\n'
              << "Saturating-add tier: "
              << simd_lab::sat_add_u8_dispatch_tier() << '\n'
              << "AXPY checksum: " << checksum << '\n'
              << "Squared error scalar: " << scalar << '\n'
              << "Squared error best:   " << best << '\n';
}
