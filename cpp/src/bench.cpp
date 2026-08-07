#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string_view>
#include <utility>
#include <vector>

namespace {
constexpr std::array<std::size_t, 6> sizes{
    1u << 10, 1u << 13, 1u << 16, 1u << 18, 1u << 20, 1u << 22};
constexpr std::size_t warmup_samples = 3;
constexpr std::size_t sample_count = 15;
constexpr std::size_t target_elements_per_sample = 1u << 20;
constexpr std::size_t max_iterations_per_sample = 4096;
constexpr std::array<std::uint16_t, 8> half_values{
    0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400};
volatile double sink = 0.0;
volatile std::uint64_t sink_u64 = 0;
volatile std::int64_t sink_i64 = 0;

struct Measurement {
    std::size_t iterations_per_sample;
    std::vector<double> ns_per_element;
};

struct Summary {
    double min;
    double median;
    double p95;
    double mad;
};

std::size_t iterations_for(std::size_t n) {
    return std::clamp(target_elements_per_sample / n,
                      std::size_t{1}, max_iterations_per_sample);
}

template <class F>
Measurement measure(std::size_t n, F&& run) {
    const auto iterations = iterations_for(n);
    for (std::size_t sample = 0; sample < warmup_samples; ++sample) {
        for (std::size_t i = 0; i < iterations; ++i) run();
    }

    std::vector<double> samples;
    samples.reserve(sample_count);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < iterations; ++i) run();
        const auto elapsed = std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - start).count();
        samples.push_back(elapsed / static_cast<double>(iterations * n));
    }
    return {iterations, std::move(samples)};
}

Summary summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto median = values[values.size() / 2];
    std::vector<double> deviations;
    deviations.reserve(values.size());
    std::transform(values.begin(), values.end(), std::back_inserter(deviations),
                   [median](double value) { return std::abs(value - median); });
    std::sort(deviations.begin(), deviations.end());
    const auto p95_index = std::min(
        values.size() - 1, (values.size() * 95 + 99) / 100 - 1);
    return {values.front(), median, values[p95_index],
            deviations[deviations.size() / 2]};
}

void report(std::string_view name, std::size_t n,
            std::size_t working_set_bytes,
            std::size_t effective_bytes_per_iteration,
            const Measurement& measurement) {
    const auto summary = summarize(measurement.ns_per_element);
    const auto bytes_per_element =
        static_cast<double>(effective_bytes_per_iteration) /
        static_cast<double>(n);
    const auto gib_s = bytes_per_element / (summary.median * 1e-9) /
                       static_cast<double>(1ULL << 30);

    std::cout << std::setprecision(9)
              << "RESULT name=" << name << " n=" << n
              << " working_set_bytes=" << working_set_bytes
              << " effective_bytes_per_iteration="
              << effective_bytes_per_iteration
              << " iterations_per_sample=" << measurement.iterations_per_sample
              << " sample_count=" << measurement.ns_per_element.size()
              << " min_ns_per_element=" << summary.min
              << " median_ns_per_element=" << summary.median
              << " p95_ns_per_element=" << summary.p95
              << " mad_ns_per_element=" << summary.mad
              << " median_gib_per_second=" << gib_s
              << " raw_ns_per_element=";
    for (std::size_t i = 0; i < measurement.ns_per_element.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << measurement.ns_per_element[i];
    }
    std::cout << '\n';
}
} // namespace

int main() {
    for (const auto n : sizes) {
        {
            std::vector<float> x(n), y(n), dst(n);
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = static_cast<float>(i) * 0.001F;
                y[i] = 1.0F + static_cast<float>(i) * 0.0005F;
            }

            auto result = measure(n, [&] {
                simd_lab::axpy_scalar(dst, x, y, 0.75F);
            });
            report("axpy/scalar-autovec", n, n * 12, n * 12, result);

            result = measure(n, [&] {
                sink = simd_lab::squared_error_scalar(x, y);
            });
            report("sqerr/scalar-f64", n, n * 8, n * 8, result);

            result = measure(n, [&] {
                sink = simd_lab::squared_error_best(x, y);
            });
            report("sqerr/best-dispatch-f64", n, n * 8, n * 8, result);
        }

        {
            std::vector<std::uint8_t> a(n), b(n);
            for (std::size_t i = 0; i < n; ++i) {
                a[i] = static_cast<std::uint8_t>((i * 17 + 3) & 255);
                b[i] = static_cast<std::uint8_t>((i * 29 + 11) & 255);
            }
            if (simd_lab::sad_u8_scalar(a, b) != simd_lab::sad_u8_best(a, b)) {
                std::cerr << "u8 SAD validation failed\n";
                return 1;
            }
            std::vector<std::uint8_t> sat_reference(n), sat_dst(n);
            simd_lab::sat_add_u8_scalar(sat_reference, a, b);
            simd_lab::sat_add_u8_best(sat_dst, a, b);
            if (sat_dst != sat_reference) {
                std::cerr << "u8 saturating-add validation failed\n";
                return 1;
            }

            auto result = measure(n, [&] {
                sink_u64 = simd_lab::sad_u8_scalar(a, b);
            });
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, result);

            result = measure(n, [&] {
                sink_u64 = simd_lab::sad_u8_best(a, b);
            });
            report("sad-u8/best-dispatch", n, n * 2, n * 2, result);

            result = measure(n, [&] {
                simd_lab::sat_add_u8_scalar(sat_dst, a, b);
            });
            report("sat-add-u8/scalar-autovec", n, n * 3, n * 3, result);

            result = measure(n, [&] {
                simd_lab::sat_add_u8_best(sat_dst, a, b);
            });
            report("sat-add-u8/best-dispatch", n, n * 3, n * 3, result);
        }

        {
            std::vector<float> f32_a(n), f32_b(n);
            std::vector<double> f64_a(n), f64_b(n);
            std::vector<std::int16_t> i16_a(n), i16_b(n);
            std::vector<std::uint8_t> u8_a(n), u8_b(n);
            std::vector<std::int8_t> i8_a(n), i8_b(n);
            std::vector<std::uint16_t> u16_a(n), u16_b(n);
            for (std::size_t i = 0; i < n; ++i) {
                f32_a[i] = static_cast<float>(i % 257) * 0.03125F - 4.0F;
                f32_b[i] = static_cast<float>(i % 193) * 0.015625F - 1.5F;
                f64_a[i] = static_cast<double>(i % 257) * 0.03125 - 4.0;
                f64_b[i] = static_cast<double>(i % 193) * 0.015625 - 1.5;
                i16_a[i] = static_cast<std::int16_t>(
                    (i * 97 + 13) & 0xffff);
                i16_b[i] = static_cast<std::int16_t>(
                    (i * 53 + 29) & 0xffff);
                u8_a[i] = static_cast<std::uint8_t>((i * 17 + 3) & 255);
                u8_b[i] = static_cast<std::uint8_t>((i * 29 + 11) & 255);
                i8_a[i] = static_cast<std::int8_t>(
                    static_cast<int>((i * 19 + 7) & 255) - 128);
                i8_b[i] = static_cast<std::int8_t>(
                    static_cast<int>((i * 23 + 5) & 255) - 128);
                u16_a[i] = static_cast<std::uint16_t>(
                    (i * 257 + 19) & 0xffff);
                u16_b[i] = static_cast<std::uint16_t>(
                    (i * 131 + 37) & 0xffff);
            }

            double f32_reference = 0.0;
            double f64_reference = 0.0;
            std::int64_t i16_dot_reference = 0;
            std::int64_t mixed_reference = 0;
            for (std::size_t i = 0; i < n; ++i) {
                f32_reference += static_cast<double>(f32_a[i]) *
                                 static_cast<double>(f32_b[i]);
                f64_reference += f64_a[i] * f64_b[i];
                i16_dot_reference += static_cast<std::int64_t>(i16_a[i]) *
                                     static_cast<std::int64_t>(i16_b[i]);
                mixed_reference += static_cast<std::int64_t>(u8_a[i]) *
                                   static_cast<std::int64_t>(i8_b[i]);
            }
            const auto close = [](double expected, double actual) {
                return std::abs(expected - actual) /
                           std::max(std::abs(expected), 1.0) <= 1e-12;
            };
            if (!close(f32_reference,
                       simd_lab::dot_f32_scalar(f32_a, f32_b)) ||
                !close(f64_reference,
                       simd_lab::dot_f64_scalar(f64_a, f64_b)) ||
                i16_dot_reference != simd_lab::dot_i16_scalar(i16_a, i16_b) ||
                mixed_reference !=
                    simd_lab::dot_u8_i8_scalar(u8_a, i8_b)) {
                std::cerr << "dot-product validation failed\n";
                return 1;
            }

            std::vector<std::uint16_t> u8_reference(n), u8_dst(n);
            std::vector<std::int16_t> i8_reference(n), i8_dst(n);
            std::vector<std::uint32_t> u16_reference(n), u16_dst(n);
            std::vector<std::int32_t> i16_product_reference(n), i16_dst(n);
            for (std::size_t i = 0; i < n; ++i) {
                const auto u8_lhs = static_cast<std::uint16_t>(u8_a[i]);
                const auto u8_rhs = static_cast<std::uint16_t>(u8_b[i]);
                u8_reference[i] =
                    static_cast<std::uint16_t>(u8_lhs * u8_rhs);
                const auto i8_lhs = static_cast<std::int16_t>(i8_a[i]);
                const auto i8_rhs = static_cast<std::int16_t>(i8_b[i]);
                i8_reference[i] =
                    static_cast<std::int16_t>(i8_lhs * i8_rhs);
                const auto u16_lhs = static_cast<std::uint32_t>(u16_a[i]);
                const auto u16_rhs = static_cast<std::uint32_t>(u16_b[i]);
                u16_reference[i] = u16_lhs * u16_rhs;
                const auto i16_lhs = static_cast<std::int32_t>(i16_a[i]);
                const auto i16_rhs = static_cast<std::int32_t>(i16_b[i]);
                i16_product_reference[i] = i16_lhs * i16_rhs;
            }
            simd_lab::widen_mul_u8_u16_scalar(u8_dst, u8_a, u8_b);
            simd_lab::widen_mul_i8_i16_scalar(i8_dst, i8_a, i8_b);
            simd_lab::widen_mul_u16_u32_scalar(
                u16_dst, u16_a, u16_b);
            simd_lab::widen_mul_i16_i32_scalar(
                i16_dst, i16_a, i16_b);
            if (u8_dst != u8_reference || i8_dst != i8_reference ||
                u16_dst != u16_reference || i16_dst != i16_product_reference) {
                std::cerr << "widening-multiply validation failed\n";
                return 1;
            }

            auto result = measure(n, [&] {
                sink = simd_lab::dot_f32_scalar(f32_a, f32_b);
            });
            report("dot-f32/scalar-f64", n, n * 8, n * 8, result);
            result = measure(n, [&] {
                sink = simd_lab::dot_f64_scalar(f64_a, f64_b);
            });
            report("dot-f64/scalar-f64", n, n * 16, n * 16, result);
            result = measure(n, [&] {
                sink_i64 = simd_lab::dot_i16_scalar(i16_a, i16_b);
            });
            report("dot-i16/scalar-i64", n, n * 4, n * 4, result);
            result = measure(n, [&] {
                sink_i64 = simd_lab::dot_u8_i8_scalar(u8_a, i8_b);
            });
            report("dot-u8-i8/scalar-i64", n, n * 2, n * 2, result);
            result = measure(n, [&] {
                simd_lab::widen_mul_u8_u16_scalar(u8_dst, u8_a, u8_b);
                sink_u64 = u8_dst[n - 1];
            });
            report("widen-mul-u8-u16/scalar-autovec", n, n * 4, n * 4,
                   result);
            result = measure(n, [&] {
                simd_lab::widen_mul_i8_i16_scalar(i8_dst, i8_a, i8_b);
                sink_i64 = i8_dst[n - 1];
            });
            report("widen-mul-i8-i16/scalar-autovec", n, n * 4, n * 4,
                   result);
            result = measure(n, [&] {
                simd_lab::widen_mul_u16_u32_scalar(
                    u16_dst, u16_a, u16_b);
                sink_u64 = u16_dst[n - 1];
            });
            report("widen-mul-u16-u32/scalar-autovec", n, n * 8, n * 8,
                   result);
            result = measure(n, [&] {
                simd_lab::widen_mul_i16_i32_scalar(
                    i16_dst, i16_a, i16_b);
                sink_i64 = i16_dst[n - 1];
            });
            report("widen-mul-i16-i32/scalar-autovec", n, n * 8, n * 8,
                   result);
        }

        {
            std::vector<std::uint16_t> c(n), lo(n, 0x3800), hi(n, 0x4000),
                dst(n);
            for (std::size_t i = 0; i < n; ++i) c[i] = half_values[i & 7];

            if (simd_lab::clamp_f16c(
                    dst.data(), c.data(), lo.data(), hi.data(), n)) {
                for (std::size_t i = 0; i < n; ++i) {
                    if (dst[i] != std::clamp(c[i], std::uint16_t{0x3800},
                                            std::uint16_t{0x4000})) {
                        std::cerr << "F16C validation failed at " << i << '\n';
                        return 1;
                    }
                }
                const auto result = measure(n, [&] {
                    if (!simd_lab::clamp_f16c(
                            dst.data(), c.data(), lo.data(), hi.data(), n)) {
                        std::abort();
                    }
                });
                report("clamp-f16/f16c-f32", n, n * 8, n * 8, result);
            } else {
                std::cout << "SKIP name=clamp-f16/f16c-f32 n=" << n
                          << " reason=avx-f16c-unavailable\n";
            }
        }
    }

    std::cout << "META size_count=" << sizes.size()
              << " warmup_samples=" << warmup_samples
              << " sample_count=" << sample_count
              << " target_elements_per_sample=" << target_elements_per_sample
              << " dispatch_tier=" << simd_lab::dispatch_tier()
              << " sat_add_u8_dispatch_tier="
              << simd_lab::sat_add_u8_dispatch_tier() << '\n';
}
