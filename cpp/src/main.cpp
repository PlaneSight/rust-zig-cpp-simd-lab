#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <limits>

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
bool run_new_operation_case(std::size_t length, XorShift64& rng,
                            bool extrema) {
    constexpr std::array<float, 8> f32_values{
        -3.0F, -1.0F, -0.0F, 0.0F, 1.0F, 2.5F,
        std::numeric_limits<float>::max(), 0.5F};
    constexpr std::array<double, 8> f64_values{
        -1.0e100, -1.0, -0.0, 0.0, 1.0, 3.5, 1.0e100, -2.25};
    constexpr std::array<std::int16_t, 8> i16_values{
        std::numeric_limits<std::int16_t>::min(), -32767, -1, 0,
        1, 32766, std::numeric_limits<std::int16_t>::max(), -12345};
    constexpr std::array<std::uint8_t, 8> u8_values{
        0, 1, 127, 128, 254, 255, 3, 200};
    constexpr std::array<std::int8_t, 8> i8_values{
        -128, -127, -1, 0, 1, 126, 127, -64};

    std::vector<float> f32_a(length), f32_b(length);
    std::vector<double> f64_a(length), f64_b(length);
    std::vector<std::int16_t> i16_a(length), i16_b(length);
    std::vector<std::uint8_t> u8_a(length), u8_b(length);
    std::vector<std::int8_t> i8_a(length), i8_b(length);
    std::vector<std::uint16_t> u16_a(length), u16_b(length);
    for (std::size_t i = 0; i < length; ++i) {
        if (extrema) {
            f32_a[i] = f32_values[i & 7];
            f32_b[i] = f32_values[(i + 3) & 7];
            f64_a[i] = f64_values[i & 7];
            f64_b[i] = f64_values[(i + 5) & 7];
            i16_a[i] = i16_values[i & 7];
            i16_b[i] = i16_values[(i + 3) & 7];
            u8_a[i] = u8_values[i & 7];
            u8_b[i] = u8_values[(i + 5) & 7];
            i8_a[i] = i8_values[i & 7];
            i8_b[i] = i8_values[(i + 2) & 7];
            u16_a[i] = (i & 1) == 0 ? 0U : 65535U;
            u16_b[i] = (i & 1) == 0 ? 65535U : 1U;
        } else {
            f32_a[i] = rng.next_float();
            f32_b[i] = rng.next_float();
            f64_a[i] = static_cast<double>(rng.next_float());
            f64_b[i] = static_cast<double>(rng.next_float());
            i16_a[i] = static_cast<std::int16_t>(rng.next());
            i16_b[i] = static_cast<std::int16_t>(rng.next());
            u8_a[i] = static_cast<std::uint8_t>(rng.next());
            u8_b[i] = static_cast<std::uint8_t>(rng.next());
            i8_a[i] = static_cast<std::int8_t>(rng.next());
            i8_b[i] = static_cast<std::int8_t>(rng.next());
            u16_a[i] = static_cast<std::uint16_t>(rng.next());
            u16_b[i] = static_cast<std::uint16_t>(rng.next());
        }
    }

    double f32_expected = 0.0;
    double f64_expected = 0.0;
    std::int64_t i16_dot_expected = 0;
    std::int64_t mixed_dot_expected = 0;
    for (std::size_t i = 0; i < length; ++i) {
        f32_expected += static_cast<double>(f32_a[i]) *
                        static_cast<double>(f32_b[i]);
        f64_expected += f64_a[i] * f64_b[i];
        i16_dot_expected += static_cast<std::int64_t>(i16_a[i]) *
                            static_cast<std::int64_t>(i16_b[i]);
        mixed_dot_expected += static_cast<std::int64_t>(u8_a[i]) *
                              static_cast<std::int64_t>(i8_b[i]);
    }
    const auto close = [](double expected, double actual) {
        return std::abs(expected - actual) /
                   std::max(std::abs(expected), 1.0) <= 1e-12;
    };
    if (!close(f32_expected, simd_lab::dot_f32_scalar(f32_a, f32_b)) ||
        !close(f64_expected, simd_lab::dot_f64_scalar(f64_a, f64_b)) ||
        i16_dot_expected != simd_lab::dot_i16_scalar(i16_a, i16_b) ||
        mixed_dot_expected != simd_lab::dot_u8_i8_scalar(u8_a, i8_b)) {
        std::cerr << "dot mismatch at length " << length << '\n';
        return false;
    }

    std::vector<std::uint16_t> u16_expected(length), u16_actual(length);
    std::vector<std::int16_t> i16_expected(length), i16_actual(length);
    std::vector<std::uint32_t> u32_expected(length), u32_actual(length);
    std::vector<std::int32_t> i32_expected(length), i32_actual(length);
    for (std::size_t i = 0; i < length; ++i) {
        const auto u8_lhs = static_cast<std::uint16_t>(u8_a[i]);
        const auto u8_rhs = static_cast<std::uint16_t>(u8_b[i]);
        u16_expected[i] = static_cast<std::uint16_t>(u8_lhs * u8_rhs);
        const auto i8_lhs = static_cast<std::int16_t>(i8_a[i]);
        const auto i8_rhs = static_cast<std::int16_t>(i8_b[i]);
        i16_expected[i] = static_cast<std::int16_t>(i8_lhs * i8_rhs);
        const auto u16_lhs = static_cast<std::uint32_t>(u16_a[i]);
        const auto u16_rhs = static_cast<std::uint32_t>(u16_b[i]);
        u32_expected[i] = u16_lhs * u16_rhs;
        const auto i16_lhs = static_cast<std::int32_t>(i16_a[i]);
        const auto i16_rhs = static_cast<std::int32_t>(i16_b[i]);
        i32_expected[i] = i16_lhs * i16_rhs;
    }
    simd_lab::widen_mul_u8_u16_scalar(u16_actual, u8_a, u8_b);
    simd_lab::widen_mul_i8_i16_scalar(i16_actual, i8_a, i8_b);
    simd_lab::widen_mul_u16_u32_scalar(u32_actual, u16_a, u16_b);
    simd_lab::widen_mul_i16_i32_scalar(i32_actual, i16_a, i16_b);
    if (u16_actual != u16_expected || i16_actual != i16_expected ||
        u32_actual != u32_expected || i32_actual != i32_expected) {
        std::cerr << "widening multiply mismatch at length " << length << '\n';
        return false;
    }
    return true;
}

bool run_randomized_differential_tests() {
    constexpr std::array<std::size_t, 13> pathological_lengths{
        0, 1, 7, 8, 9, 15, 16, 31, 32, 63, 64, 65, 127};
    constexpr std::array<std::uint16_t, 8> half_values{
        0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400};
    XorShift64 rng{0x8f3ca516d27b49e1ULL};
    for (const auto length : pathological_lengths) {
        if (!run_new_operation_case(length, rng, true)) return false;
    }

    for (std::size_t trial = 0; trial < 256; ++trial) {
        const std::size_t length = trial < 64 ? trial : rng.next() % 2050;
        if (!run_new_operation_case(length, rng, false)) return false;
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
