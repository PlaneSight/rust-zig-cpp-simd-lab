#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <iterator>
#include <string_view>
#include <type_traits>
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

template <typename T>
T sat_add_reference(T lhs, T rhs) noexcept {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_unsigned_v<T>) {
        constexpr T max_value = std::numeric_limits<T>::max();
        return lhs > static_cast<T>(max_value - rhs)
                   ? max_value
                   : static_cast<T>(lhs + rhs);
    } else {
        constexpr T min_value = std::numeric_limits<T>::min();
        constexpr T max_value = std::numeric_limits<T>::max();
        if ((rhs > 0 && lhs > static_cast<T>(max_value - rhs)) ||
            (rhs < 0 && lhs < static_cast<T>(min_value - rhs))) {
            return rhs > 0 ? max_value : min_value;
        }
        return static_cast<T>(lhs + rhs);
    }
}

template <typename T>
T sat_sub_reference(T lhs, T rhs) noexcept {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_unsigned_v<T>) {
        const auto widened_lhs = static_cast<std::uint64_t>(lhs);
        const auto widened_rhs = static_cast<std::uint64_t>(rhs);
        return widened_lhs < widened_rhs
                   ? T{0}
                   : static_cast<T>(widened_lhs - widened_rhs);
    } else if constexpr (sizeof(T) < sizeof(std::int64_t)) {
        const auto difference = static_cast<std::int64_t>(lhs) -
                                static_cast<std::int64_t>(rhs);
        const auto min_value =
            static_cast<std::int64_t>(std::numeric_limits<T>::min());
        const auto max_value =
            static_cast<std::int64_t>(std::numeric_limits<T>::max());
        return static_cast<T>(
            std::clamp(difference, min_value, max_value));
    } else {
        constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
        constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
        if (rhs > 0 && lhs < min_value + rhs) return min_value;
        if (rhs < 0 && lhs > max_value + rhs) return max_value;
        return lhs - rhs;
    }
}


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

template <typename T, typename F>
bool run_sat_add_case(std::string_view name, std::size_t n,
                      const std::vector<T>& a, const std::vector<T>& b,
                      F function, std::size_t bytes_per_element) {
    std::vector<T> expected(n), dst(n);
    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = sat_add_reference(a[i], b[i]);
    }
    function(dst, a, b);
    if (dst != expected) {
        std::cerr << name << " validation failed\n";
        return false;
    }

    const auto result = measure(n, [&] {
        function(dst, a, b);
        if constexpr (std::is_signed_v<T>) {
            sink_i64 = static_cast<std::int64_t>(dst[n - 1]);
        } else {
            sink_u64 = static_cast<std::uint64_t>(dst[n - 1]);
        }
    });
    report(name, n, n * bytes_per_element, n * bytes_per_element, result);
    return true;
}

template <typename T, typename F>
bool run_sat_sub_case(std::string_view name, std::size_t n,
                      const std::vector<T>& a, const std::vector<T>& b,
                      F function, std::size_t bytes_per_element) {
    std::vector<T> expected(n), dst(n);
    for (std::size_t i = 0; i < n; ++i) {
        expected[i] = sat_sub_reference(a[i], b[i]);
    }
    function(dst, a, b);
    if (dst != expected) {
        std::cerr << name << " validation failed\n";
        return false;
    }

    const auto result = measure(n, [&] {
        function(dst, a, b);
        if constexpr (std::is_signed_v<T>) {
            sink_i64 = static_cast<std::int64_t>(dst[n - 1]);
        } else {
            sink_u64 = static_cast<std::uint64_t>(dst[n - 1]);
        }
    });
    report(name, n, n * bytes_per_element, n * bytes_per_element, result);
    return true;
}


std::uint8_t blend_u8_bench_reference(std::uint8_t a, std::uint8_t b,
                                      std::uint16_t weight) noexcept {
    const auto weighted_b = static_cast<std::uint32_t>(weight);
    const auto weighted_a = 256U - weighted_b;
    const auto sum =
        static_cast<std::uint32_t>(a) * weighted_a +
        static_cast<std::uint32_t>(b) * weighted_b + 128U;
    return static_cast<std::uint8_t>(sum >> 8U);
}

std::uint8_t convolve3_u8_bench_reference(
    const std::vector<std::uint8_t>& src, std::size_t i) noexcept {
    const auto left = i == 0U ? 0U : i - 1U;
    const auto right = i + 1U < src.size() ? i + 1U : src.size() - 1U;
    const auto sum =
        static_cast<std::uint32_t>(src[left]) +
        2U * static_cast<std::uint32_t>(src[i]) +
        static_cast<std::uint32_t>(src[right]) + 2U;
    return static_cast<std::uint8_t>(sum >> 2U);
}

std::uint8_t convolve5_u8_bench_reference(
    const std::vector<std::uint8_t>& src, std::size_t i) noexcept {
    const auto sample = [&src](std::int64_t index) {
        if (index < 0) return static_cast<std::uint32_t>(src.front());
        const auto unsigned_index = static_cast<std::size_t>(index);
        return static_cast<std::uint32_t>(
            src[std::min(unsigned_index, src.size() - 1U)]);
    };
    const auto center = static_cast<std::int64_t>(i);
    const auto sum =
        sample(center - 2) + 4U * sample(center - 1) +
        6U * sample(center) + 4U * sample(center + 1) +
        sample(center + 2) + 8U;
    return static_cast<std::uint8_t>(sum >> 4U);
}

bool run_image_kernel_benchmarks(std::size_t n) {
    if (n == 0U) return true;
    constexpr std::uint16_t weight = 77;
    std::vector<std::uint8_t> a(n), b(n), src(n);
    std::vector<std::uint8_t> blend_expected(n), convolve3_expected(n),
        convolve5_expected(n);
    std::vector<std::uint8_t> blend_dst(n), convolve3_dst(n),
        convolve5_dst(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::uint8_t>((i * 29U + 11U) & 255U);
        b[i] = static_cast<std::uint8_t>((i * 47U + 23U) & 255U);
        src[i] = static_cast<std::uint8_t>((i * 73U + 5U) & 255U);
    }
    for (std::size_t i = 0; i < n; ++i) {
        blend_expected[i] = blend_u8_bench_reference(a[i], b[i], weight);
        convolve3_expected[i] = convolve3_u8_bench_reference(src, i);
        convolve5_expected[i] = convolve5_u8_bench_reference(src, i);
    }

    simd_lab::blend_u8_scalar(blend_dst, a, b, weight);
    if (blend_dst != blend_expected) {
        std::cerr << "blend-u8/scalar-autovec validation failed\n";
        return false;
    }
    simd_lab::convolve3_u8_scalar(convolve3_dst, src);
    if (convolve3_dst != convolve3_expected) {
        std::cerr << "convolve3-u8/scalar-autovec validation failed\n";
        return false;
    }
    simd_lab::convolve5_u8_scalar(convolve5_dst, src);
    if (convolve5_dst != convolve5_expected) {
        std::cerr << "convolve5-u8/scalar-autovec validation failed\n";
        return false;
    }

    auto result = measure(n, [&] {
        simd_lab::blend_u8_scalar(blend_dst, a, b, weight);
        sink_u64 = blend_dst[n - 1U];
    });
    report("blend-u8/scalar-autovec", n, n * 3U, n * 3U, result);

    result = measure(n, [&] {
        simd_lab::convolve3_u8_scalar(convolve3_dst, src);
        sink_u64 = convolve3_dst[n - 1U];
    });
    report("convolve3-u8/scalar-autovec", n, n * 2U, n * 2U, result);

    result = measure(n, [&] {
        simd_lab::convolve5_u8_scalar(convolve5_dst, src);
        sink_u64 = convolve5_dst[n - 1U];
    });
    report("convolve5-u8/scalar-autovec", n, n * 2U, n * 2U, result);
    return true;
}


bool run_mixed_width_benchmarks(std::size_t n) {
    if (n == 0) return true;

    std::vector<std::uint8_t> u8(n);
    std::vector<std::int8_t> i8(n);
    std::vector<std::int16_t> i16(n);
    std::vector<std::uint16_t> u16(n);
    for (std::size_t i = 0; i < n; ++i) {
        u8[i] = static_cast<std::uint8_t>((i * 17U + 3U) & 255U);
        i8[i] = static_cast<std::int8_t>(
            static_cast<int>((i * 19U + 7U) & 255U) - 128);
        const auto i16_raw = static_cast<std::int32_t>(
            (i * 97U + 13U) & 0xffffU);
        i16[i] = static_cast<std::int16_t>(
            i16_raw < 32768 ? i16_raw : i16_raw - 65536);
        u16[i] = static_cast<std::uint16_t>(
            (i * 257U + 19U) & 0xffffU);
    }

    const auto run_unary =
        [&](std::string_view name, const auto& src, auto& dst, auto function,
            auto expected_value, std::size_t bytes_per_element) {
            using output_type =
                typename std::decay_t<decltype(dst)>::value_type;
            std::vector<output_type> expected(n);
            for (std::size_t i = 0; i < n; ++i) {
                expected[i] = expected_value(src[i]);
            }
            function(dst, src);
            if (dst != expected) {
                std::cerr << name << " validation failed\n";
                return false;
            }
            const auto result = measure(n, [&] {
                function(dst, src);
                if constexpr (std::is_floating_point_v<output_type>) {
                    sink = static_cast<double>(dst[n - 1]);
                } else if constexpr (std::is_signed_v<output_type>) {
                    sink_i64 = static_cast<std::int64_t>(dst[n - 1]);
                } else {
                    sink_u64 = static_cast<std::uint64_t>(dst[n - 1]);
                }
            });
            report(name, n,
                   n * (sizeof(typename std::decay_t<decltype(src)>::value_type) +
                        sizeof(output_type)),
                   n * bytes_per_element, result);
            return true;
        };

    std::vector<std::uint16_t> u8_to_u16(n);
    if (!run_unary(
            "widen-u8-u16/scalar-autovec", u8, u8_to_u16,
            simd_lab::widen_u8_to_u16_scalar,
            [](std::uint8_t value) {
                return static_cast<std::uint16_t>(value);
            },
            sizeof(std::uint8_t) + sizeof(std::uint16_t))) {
        return false;
    }
    std::vector<std::uint32_t> u8_to_u32(n);
    if (!run_unary(
            "widen-u8-u32/scalar-autovec", u8, u8_to_u32,
            simd_lab::widen_u8_to_u32_scalar,
            [](std::uint8_t value) {
                return static_cast<std::uint32_t>(value);
            },
            sizeof(std::uint8_t) + sizeof(std::uint32_t))) {
        return false;
    }
    std::vector<std::int16_t> i8_to_i16(n);
    if (!run_unary(
            "widen-i8-i16/scalar-autovec", i8, i8_to_i16,
            simd_lab::widen_i8_to_i16_scalar,
            [](std::int8_t value) {
                return static_cast<std::int16_t>(value);
            },
            sizeof(std::int8_t) + sizeof(std::int16_t))) {
        return false;
    }
    std::vector<std::int32_t> i16_to_i32(n);
    if (!run_unary(
            "widen-i16-i32/scalar-autovec", i16, i16_to_i32,
            simd_lab::widen_i16_to_i32_scalar,
            [](std::int16_t value) {
                return static_cast<std::int32_t>(value);
            },
            sizeof(std::int16_t) + sizeof(std::int32_t))) {
        return false;
    }
    std::vector<std::uint32_t> u16_to_u32(n);
    if (!run_unary(
            "widen-u16-u32/scalar-autovec", u16, u16_to_u32,
            simd_lab::widen_u16_to_u32_scalar,
            [](std::uint16_t value) {
                return static_cast<std::uint32_t>(value);
            },
            sizeof(std::uint16_t) + sizeof(std::uint32_t))) {
        return false;
    }

    std::vector<float> affine(n), u16_to_f32(n), i16_to_f32(n);
    constexpr float scale = 0.75F;
    constexpr float bias = -2.25F;
    if (!run_unary(
            "convert-u8-f32-affine/scalar-autovec", u8, affine,
            [=](std::span<float> dst, std::span<const std::uint8_t> src) {
                simd_lab::convert_u8_f32_affine_scalar(dst, src, scale, bias);
            },
            [=](std::uint8_t value) {
                return static_cast<float>(value) * scale + bias;
            },
            sizeof(std::uint8_t) + sizeof(float))) {
        return false;
    }
    if (!run_unary(
            "convert-u16-f32/scalar-autovec", u16, u16_to_f32,
            simd_lab::convert_u16_to_f32_scalar,
            [](std::uint16_t value) { return static_cast<float>(value); },
            sizeof(std::uint16_t) + sizeof(float)) ||
        !run_unary(
            "convert-i16-f32/scalar-autovec", i16, i16_to_f32,
            simd_lab::convert_i16_to_f32_scalar,
            [](std::int16_t value) { return static_cast<float>(value); },
            sizeof(std::int16_t) + sizeof(float))) {
        return false;
    }

    const std::array<float, 8> f32_u16_values{
        -1.0F, 0.0F, 0.5F, 1.5F, 65534.9F, 65535.0F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> f32_u16(n);
    for (std::size_t i = 0; i < n; ++i) {
        f32_u16[i] = f32_u16_values[i & 7U];
    }
    std::vector<std::uint16_t> f32_u16_dst(n);
    if (!run_unary(
            "convert-f32-u16-sat/scalar-autovec", f32_u16, f32_u16_dst,
            simd_lab::f32_to_u16_sat_scalar,
            [](float value) {
                if (!(value > 0.0F)) return std::uint16_t{0};
                if (value >= 65535.0F) {
                    return std::numeric_limits<std::uint16_t>::max();
                }
                return static_cast<std::uint16_t>(value);
            },
            sizeof(float) + sizeof(std::uint16_t))) {
        return false;
    }

    const std::array<float, 8> f32_u8_values{
        0.0F, 0.49F, 0.5F, 1.5F, 127.5F, 254.5F, 254.99F, 255.0F};
    std::vector<float> f32_u8(n);
    for (std::size_t i = 0; i < n; ++i) {
        f32_u8[i] = f32_u8_values[i & 7U];
    }
    std::vector<std::uint8_t> f32_u8_dst(n);
    if (!run_unary(
            "convert-f32-u8-trunc/scalar-autovec", f32_u8, f32_u8_dst,
            simd_lab::convert_f32_u8_trunc_scalar,
            [](float value) { return static_cast<std::uint8_t>(value); },
            sizeof(float) + sizeof(std::uint8_t)) ||
        !run_unary(
            "convert-f32-u8-round/scalar-autovec", f32_u8, f32_u8_dst,
            simd_lab::convert_f32_u8_round_scalar,
            [](float value) {
                return static_cast<std::uint8_t>(std::floor(value + 0.5F));
            },
            sizeof(float) + sizeof(std::uint8_t))) {
        return false;
    }
    if (!run_unary(
            "convert-f32-u8-sat/scalar-autovec", f32_u16, f32_u8_dst,
            simd_lab::convert_f32_u8_sat_scalar,
            [](float value) {
                if (!(value > 0.0F)) return std::uint8_t{0};
                if (value >= 255.0F) {
                    return std::numeric_limits<std::uint8_t>::max();
                }
                return static_cast<std::uint8_t>(value);
            },
            sizeof(float) + sizeof(std::uint8_t))) {
        return false;
    }

    std::vector<std::uint8_t> narrow_dst(n);
    if (!run_unary(
            "narrow-u16-u8-trunc/scalar-autovec", u16, narrow_dst,
            simd_lab::narrow_u16_to_u8_trunc_scalar,
            [](std::uint16_t value) {
                return static_cast<std::uint8_t>(value & 0xffU);
            },
            sizeof(std::uint16_t) + sizeof(std::uint8_t)) ||
        !run_unary(
            "narrow-u16-u8-round/scalar-autovec", u16, narrow_dst,
            simd_lab::narrow_u16_to_u8_round_scalar,
            [](std::uint16_t value) {
                return static_cast<std::uint8_t>(
                    (static_cast<std::uint32_t>(value) + 128U) / 257U);
            },
            sizeof(std::uint16_t) + sizeof(std::uint8_t)) ||
        !run_unary(
            "narrow-u16-u8-sat/scalar-autovec", u16, narrow_dst,
            simd_lab::narrow_u16_to_u8_sat_scalar,
            [](std::uint16_t value) {
                return static_cast<std::uint8_t>(
                    std::min<std::uint16_t>(value, 255U));
            },
            sizeof(std::uint16_t) + sizeof(std::uint8_t))) {
        return false;
    }

    std::vector<std::uint8_t> bytes(n * 4U), unpacked(n * 4U);
    std::vector<std::uint32_t> packed(n), packed_expected(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto base = i * 4U;
        bytes[base] = static_cast<std::uint8_t>((i * 13U + 1U) & 255U);
        bytes[base + 1U] =
            static_cast<std::uint8_t>((i * 17U + 2U) & 255U);
        bytes[base + 2U] =
            static_cast<std::uint8_t>((i * 19U + 3U) & 255U);
        bytes[base + 3U] =
            static_cast<std::uint8_t>((i * 23U + 4U) & 255U);
        packed_expected[i] =
            static_cast<std::uint32_t>(bytes[base]) |
            (static_cast<std::uint32_t>(bytes[base + 1U]) << 8U) |
            (static_cast<std::uint32_t>(bytes[base + 2U]) << 16U) |
            (static_cast<std::uint32_t>(bytes[base + 3U]) << 24U);
    }
    simd_lab::pack_u8x4_to_u32_scalar(packed, bytes);
    if (packed != packed_expected) {
        std::cerr << "pack-u8x4-u32/scalar-autovec validation failed\n";
        return false;
    }
    auto result = measure(n, [&] {
        simd_lab::pack_u8x4_to_u32_scalar(packed, bytes);
        sink_u64 = packed[n - 1];
    });
    report("pack-u8x4-u32/scalar-autovec", n, n * 8U, n * 8U, result);

    simd_lab::unpack_u32_to_u8x4_scalar(unpacked, packed);
    if (unpacked != bytes) {
        std::cerr << "unpack-u32-u8x4/scalar-autovec validation failed\n";
        return false;
    }
    result = measure(n, [&] {
        simd_lab::unpack_u32_to_u8x4_scalar(unpacked, packed);
        sink_u64 = unpacked[n * 4U - 1U];
    });
    report("unpack-u32-u8x4/scalar-autovec", n, n * 8U, n * 8U, result);
    return true;
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
            std::vector<std::uint16_t> u16_a(n), u16_b(n);
            for (std::size_t i = 0; i < n; ++i) {
                a[i] = static_cast<std::uint8_t>((i * 17 + 3) & 255);
                b[i] = static_cast<std::uint8_t>((i * 29 + 11) & 255);
                u16_a[i] =
                    static_cast<std::uint16_t>((i * 257 + 3) & 0xffffU);
                u16_b[i] =
                    static_cast<std::uint16_t>((i * 911 + 11) & 0xffffU);
            }
            if (simd_lab::sad_u8_scalar(a, b) != simd_lab::sad_u8_best(a, b)) {
                std::cerr << "u8 SAD validation failed\n";
                return 1;
            }
            if (simd_lab::sad_u16_scalar(u16_a, u16_b) !=
                simd_lab::sad_u16_best(u16_a, u16_b)) {
                std::cerr << "u16 SAD validation failed\n";
                return 1;
            }

            std::vector<std::uint8_t> sat_reference(n), sat_dst(n);
            simd_lab::sat_add_u8_scalar(sat_reference, a, b);
            simd_lab::sat_add_u8_best(sat_dst, a, b);
            if (sat_dst != sat_reference) {
                std::cerr << "u8 saturating-add validation failed\n";
                return 1;
            }

            std::vector<std::uint8_t> sat_sub_expected(n), sat_sub_dst(n);
            for (std::size_t i = 0; i < n; ++i) {
                sat_sub_expected[i] = sat_sub_reference(a[i], b[i]);
            }
            simd_lab::sat_sub_u8_scalar(sat_sub_dst, a, b);
            if (sat_sub_dst != sat_sub_expected) {
                std::cerr << "u8 saturating-sub scalar validation failed\n";
                return 1;
            }
            simd_lab::sat_sub_u8_best(sat_sub_dst, a, b);
            if (sat_sub_dst != sat_sub_expected) {
                std::cerr << "u8 saturating-sub best validation failed\n";
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
                sink_u64 = simd_lab::sad_u16_scalar(u16_a, u16_b);
            });
            report("sad-u16/scalar-autovec", n, n * 4, n * 4, result);

            result = measure(n, [&] {
                sink_u64 = simd_lab::sad_u16_best(u16_a, u16_b);
            });
            report("sad-u16/best-dispatch", n, n * 4, n * 4, result);

            result = measure(n, [&] {
                simd_lab::sat_add_u8_scalar(sat_dst, a, b);
            });
            report("sat-add-u8/scalar-autovec", n, n * 3, n * 3, result);

            result = measure(n, [&] {
                simd_lab::sat_add_u8_best(sat_dst, a, b);
            });
            report("sat-add-u8/best-dispatch", n, n * 3, n * 3, result);

            result = measure(n, [&] {
                simd_lab::sat_sub_u8_scalar(sat_sub_dst, a, b);
                sink_u64 = sat_sub_dst[n - 1U];
            });
            report("sat-sub-u8/scalar-autovec", n,
                   n * sizeof(std::uint8_t) * 3U,
                   n * sizeof(std::uint8_t) * 3U, result);

            result = measure(n, [&] {
                simd_lab::sat_sub_u8_best(sat_sub_dst, a, b);
                sink_u64 = sat_sub_dst[n - 1U];
            });
            report("sat-sub-u8/best-dispatch", n,
                   n * sizeof(std::uint8_t) * 3U,
                   n * sizeof(std::uint8_t) * 3U, result);
        }

        {
            std::vector<float> f32_a(n), f32_b(n);
            std::vector<double> f64_a(n), f64_b(n);
            std::vector<std::int16_t> i16_a(n), i16_b(n);
            std::vector<std::uint8_t> u8_a(n), u8_b(n);
            std::vector<std::int8_t> i8_a(n), i8_b(n);
            std::vector<std::uint16_t> u16_a(n), u16_b(n);
            std::vector<std::uint32_t> u32_a(n), u32_b(n);
            std::vector<std::int32_t> i32_a(n), i32_b(n);
            std::vector<std::uint64_t> u64_a(n), u64_b(n);
            std::vector<std::int64_t> i64_a(n), i64_b(n);
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
                u32_a[i] = static_cast<std::uint32_t>(
                    i * 2654435761ULL + 17U);
                u32_b[i] = static_cast<std::uint32_t>(
                    i * 2246822519ULL + 31U);
                i32_a[i] = static_cast<std::int32_t>(
                    static_cast<std::int64_t>(i % 2000001) - 1000000);
                i32_b[i] = static_cast<std::int32_t>(
                    static_cast<std::int64_t>(i % 1600001) - 800000);
                u64_a[i] = i * 11400714819323198485ULL + 23U;
                u64_b[i] = i * 7046029254386353131ULL + 41U;
                i64_a[i] = static_cast<std::int64_t>(i % 2000001) - 1000000;
                i64_b[i] = static_cast<std::int64_t>(i % 1600001) - 800000;
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
            std::vector<std::uint64_t> u32_product_reference(n),
                u32_product_dst(n);
            std::vector<std::int64_t> i32_product_reference(n),
                i32_product_dst(n);
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
                const auto u32_lhs = static_cast<std::uint64_t>(u32_a[i]);
                const auto u32_rhs = static_cast<std::uint64_t>(u32_b[i]);
                u32_product_reference[i] = u32_lhs * u32_rhs;
                const auto i32_lhs = static_cast<std::int64_t>(i32_a[i]);
                const auto i32_rhs = static_cast<std::int64_t>(i32_b[i]);
                i32_product_reference[i] = i32_lhs * i32_rhs;
            }
            simd_lab::widen_mul_u8_u16_scalar(u8_dst, u8_a, u8_b);
            simd_lab::widen_mul_i8_i16_scalar(i8_dst, i8_a, i8_b);
            simd_lab::widen_mul_u16_u32_scalar(
                u16_dst, u16_a, u16_b);
            simd_lab::widen_mul_i16_i32_scalar(
                i16_dst, i16_a, i16_b);
            simd_lab::widen_mul_u32_u64_scalar(
                u32_product_dst, u32_a, u32_b);
            simd_lab::widen_mul_i32_i64_scalar(
                i32_product_dst, i32_a, i32_b);
            if (u8_dst != u8_reference || i8_dst != i8_reference ||
                u16_dst != u16_reference ||
                i16_dst != i16_product_reference ||
                u32_product_dst != u32_product_reference ||
                i32_product_dst != i32_product_reference) {
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
            result = measure(n, [&] {
                simd_lab::widen_mul_u32_u64_scalar(
                    u32_product_dst, u32_a, u32_b);
                sink_u64 = u32_product_dst[n - 1];
            });
            report("widen-mul-u32-u64/scalar-autovec", n, n * 16, n * 16,
                   result);
            result = measure(n, [&] {
                simd_lab::widen_mul_i32_i64_scalar(
                    i32_product_dst, i32_a, i32_b);
                sink_i64 = i32_product_dst[n - 1];
            });
            report("widen-mul-i32-i64/scalar-autovec", n, n * 16, n * 16,
                   result);
            if (!run_image_kernel_benchmarks(n)) {
                return 1;
            }
            if (!run_mixed_width_benchmarks(n)) {
                return 1;
            }
            if (!run_sat_add_case("sat-add-i8/scalar-autovec", n, i8_a, i8_b,
                                  simd_lab::sat_add_i8_scalar,
                                  sizeof(std::int8_t) * 3) ||
                !run_sat_add_case("sat-add-u16/scalar-autovec", n, u16_a, u16_b,
                                  simd_lab::sat_add_u16_scalar,
                                  sizeof(std::uint16_t) * 3) ||
                !run_sat_add_case("sat-add-i16/scalar-autovec", n, i16_a, i16_b,
                                  simd_lab::sat_add_i16_scalar,
                                  sizeof(std::int16_t) * 3) ||
                !run_sat_add_case("sat-add-u32/scalar-autovec", n, u32_a, u32_b,
                                  simd_lab::sat_add_u32_scalar,
                                  sizeof(std::uint32_t) * 3) ||
                !run_sat_add_case("sat-add-i32/scalar-autovec", n, i32_a, i32_b,
                                  simd_lab::sat_add_i32_scalar,
                                  sizeof(std::int32_t) * 3) ||
                !run_sat_add_case("sat-add-u64/scalar-autovec", n, u64_a, u64_b,
                                  simd_lab::sat_add_u64_scalar,
                                  sizeof(std::uint64_t) * 3) ||
                !run_sat_add_case("sat-add-i64/scalar-autovec", n, i64_a, i64_b,
                                  simd_lab::sat_add_i64_scalar,
                                  sizeof(std::int64_t) * 3)) {
                return 1;
            }
            if (!run_sat_sub_case("sat-sub-i8/scalar-autovec", n, i8_a, i8_b,
                                  simd_lab::sat_sub_i8_scalar,
                                  sizeof(std::int8_t) * 3) ||
                !run_sat_sub_case("sat-sub-u16/scalar-autovec", n, u16_a,
                                  u16_b, simd_lab::sat_sub_u16_scalar,
                                  sizeof(std::uint16_t) * 3) ||
                !run_sat_sub_case("sat-sub-i16/scalar-autovec", n, i16_a,
                                  i16_b, simd_lab::sat_sub_i16_scalar,
                                  sizeof(std::int16_t) * 3) ||
                !run_sat_sub_case("sat-sub-u32/scalar-autovec", n, u32_a,
                                  u32_b, simd_lab::sat_sub_u32_scalar,
                                  sizeof(std::uint32_t) * 3) ||
                !run_sat_sub_case("sat-sub-i32/scalar-autovec", n, i32_a,
                                  i32_b, simd_lab::sat_sub_i32_scalar,
                                  sizeof(std::int32_t) * 3) ||
                !run_sat_sub_case("sat-sub-u64/scalar-autovec", n, u64_a,
                                  u64_b, simd_lab::sat_sub_u64_scalar,
                                  sizeof(std::uint64_t) * 3) ||
                !run_sat_sub_case("sat-sub-i64/scalar-autovec", n, i64_a,
                                  i64_b, simd_lab::sat_sub_i64_scalar,
                                  sizeof(std::int64_t) * 3)) {
                return 1;
            }


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
              << simd_lab::sat_add_u8_dispatch_tier()
              << " sat_sub_u8_dispatch_tier="
              << simd_lab::sat_sub_u8_dispatch_tier() << '\n';
}
