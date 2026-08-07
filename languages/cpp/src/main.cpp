#include "kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <limits>

#include <type_traits>
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
    constexpr std::array<std::uint32_t, 8> u32_values{
        0U, 1U, 0x7fffffffU, 0x80000000U,
        std::numeric_limits<std::uint32_t>::max(), 3U,
        0xaaaaaaaaU, 0x55555555U};
    constexpr std::array<std::int32_t, 8> i32_values{
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::min() + 1,
        -1, 0, 1, std::numeric_limits<std::int32_t>::max() - 1,
        std::numeric_limits<std::int32_t>::max(), -123456789};

    std::vector<float> f32_a(length), f32_b(length);
    std::vector<double> f64_a(length), f64_b(length);
    std::vector<std::int16_t> i16_a(length), i16_b(length);
    std::vector<std::uint8_t> u8_a(length), u8_b(length);
    std::vector<std::int8_t> i8_a(length), i8_b(length);
    std::vector<std::uint16_t> u16_a(length), u16_b(length);
    std::vector<std::uint32_t> u32_a(length), u32_b(length);
    std::vector<std::int32_t> i32_a(length), i32_b(length);
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
            i8_b[i] = i8_values[(i + 3) & 7];
            u16_a[i] = (i & 1) == 0 ? 0U : 65535U;
            u16_b[i] = (i & 1) == 0 ? 65535U : 0U;
            u32_a[i] = u32_values[i & 7];
            u32_b[i] = u32_values[(i + 5) & 7];
            i32_a[i] = i32_values[i & 7];
            i32_b[i] = i32_values[(i + 3) & 7];
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
            u32_a[i] = static_cast<std::uint32_t>(rng.next());
            u32_b[i] = static_cast<std::uint32_t>(rng.next());
            i32_a[i] = static_cast<std::int32_t>(rng.next());
            i32_b[i] = static_cast<std::int32_t>(rng.next());
        }
    }
    double f32_expected = 0.0;
    double f64_expected = 0.0;
    std::int64_t i16_dot_expected = 0;
    std::int64_t mixed_dot_expected = 0;
    std::uint64_t sad_u16_expected = 0;
    for (std::size_t i = 0; i < length; ++i) {
        f32_expected += static_cast<double>(f32_a[i]) *
                        static_cast<double>(f32_b[i]);
        f64_expected += f64_a[i] * f64_b[i];
        i16_dot_expected += static_cast<std::int64_t>(i16_a[i]) *
                            static_cast<std::int64_t>(i16_b[i]);
        mixed_dot_expected += static_cast<std::int64_t>(u8_a[i]) *
                              static_cast<std::int64_t>(i8_b[i]);
        const auto u16_lhs = static_cast<std::uint32_t>(u16_a[i]);
        const auto u16_rhs = static_cast<std::uint32_t>(u16_b[i]);
        sad_u16_expected +=
            u16_lhs > u16_rhs ? static_cast<std::uint64_t>(u16_lhs - u16_rhs)
                              : static_cast<std::uint64_t>(u16_rhs - u16_lhs);
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
    if (sad_u16_expected != simd_lab::sad_u16_scalar(u16_a, u16_b) ||
        sad_u16_expected != simd_lab::sad_u16_best(u16_a, u16_b)) {
        std::cerr << "u16 SAD mismatch at length " << length << '\n';
        return false;
    }

    std::vector<std::uint16_t> u16_expected(length), u16_actual(length);
    std::vector<std::int16_t> i16_expected(length), i16_actual(length);
    std::vector<std::uint32_t> u32_expected(length), u32_actual(length);
    std::vector<std::int32_t> i32_expected(length), i32_actual(length);
    std::vector<std::uint64_t> u64_expected(length), u64_actual(length);
    std::vector<std::int64_t> i64_expected(length), i64_actual(length);
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
        const auto u32_lhs = static_cast<std::uint64_t>(u32_a[i]);
        const auto u32_rhs = static_cast<std::uint64_t>(u32_b[i]);
        u64_expected[i] = u32_lhs * u32_rhs;
        const auto i32_lhs = static_cast<std::int64_t>(i32_a[i]);
        const auto i32_rhs = static_cast<std::int64_t>(i32_b[i]);
        i64_expected[i] = i32_lhs * i32_rhs;
    }
    simd_lab::widen_mul_u8_u16_scalar(u16_actual, u8_a, u8_b);
    simd_lab::widen_mul_i8_i16_scalar(i16_actual, i8_a, i8_b);
    simd_lab::widen_mul_u16_u32_scalar(u32_actual, u16_a, u16_b);
    simd_lab::widen_mul_i16_i32_scalar(i32_actual, i16_a, i16_b);
    simd_lab::widen_mul_u32_u64_scalar(u64_actual, u32_a, u32_b);
    simd_lab::widen_mul_i32_i64_scalar(i64_actual, i32_a, i32_b);
    if (u16_actual != u16_expected || i16_actual != i16_expected ||
        u32_actual != u32_expected || i32_actual != i32_expected ||
        u64_actual != u64_expected || i64_actual != i64_expected) {
        std::cerr << "widening multiply mismatch at length " << length << '\n';
        return false;
    }
    return true;
}
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


bool run_saturating_add_case(std::size_t length, XorShift64& rng,
                             bool extrema) {
    constexpr std::array<std::int8_t, 8> i8_a_values{
        std::numeric_limits<std::int8_t>::min(), -127, -1, 0,
        1, 126, std::numeric_limits<std::int8_t>::max(), 42};
    constexpr std::array<std::int8_t, 8> i8_b_values{
        -1, -1, std::numeric_limits<std::int8_t>::min(), 0,
        1, 1, std::numeric_limits<std::int8_t>::max(), -42};
    constexpr std::array<std::uint16_t, 8> u16_a_values{
        0, 1, 2, std::numeric_limits<std::uint16_t>::max() / 2,
        std::numeric_limits<std::uint16_t>::max() - 1,
        std::numeric_limits<std::uint16_t>::max(), 0x8000, 17};
    constexpr std::array<std::uint16_t, 8> u16_b_values{
        0, 1, std::numeric_limits<std::uint16_t>::max(),
        std::numeric_limits<std::uint16_t>::max() / 2,
        std::numeric_limits<std::uint16_t>::max() - 1, 1, 0x8000,
        std::numeric_limits<std::uint16_t>::max()};
    constexpr std::array<std::int16_t, 8> i16_a_values{
        std::numeric_limits<std::int16_t>::min(),
        std::numeric_limits<std::int16_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int16_t>::max() - 1,
        std::numeric_limits<std::int16_t>::max(), 12345};
    constexpr std::array<std::int16_t, 8> i16_b_values{
        -1, -1, std::numeric_limits<std::int16_t>::min(), 0, 1, 1,
        std::numeric_limits<std::int16_t>::max(), -12345};
    constexpr std::array<std::uint32_t, 8> u32_a_values{
        0U, 1U, 2U, std::numeric_limits<std::uint32_t>::max() / 2,
        std::numeric_limits<std::uint32_t>::max() - 1,
        std::numeric_limits<std::uint32_t>::max(), 0x80000000U,
        0x01234567U};
    constexpr std::array<std::uint32_t, 8> u32_b_values{
        0U, 1U, std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max() / 2,
        std::numeric_limits<std::uint32_t>::max() - 1, 1U,
        0x80000000U, std::numeric_limits<std::uint32_t>::max()};
    constexpr std::array<std::int32_t, 8> i32_a_values{
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int32_t>::max() - 1,
        std::numeric_limits<std::int32_t>::max(), 123456789};
    constexpr std::array<std::int32_t, 8> i32_b_values{
        -1, -1, std::numeric_limits<std::int32_t>::min(), 0, 1, 1,
        std::numeric_limits<std::int32_t>::max(), -123456789};
    constexpr std::array<std::uint64_t, 8> u64_a_values{
        0, 1, 2, std::numeric_limits<std::uint64_t>::max() / 2,
        std::numeric_limits<std::uint64_t>::max() - 1,
        std::numeric_limits<std::uint64_t>::max(),
        std::uint64_t{1} << 63, 0x0123456789abcdefULL};
    constexpr std::array<std::uint64_t, 8> u64_b_values{
        0, std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max() / 2,
        std::numeric_limits<std::uint64_t>::max() - 1, 1,
        std::uint64_t{1} << 63, std::numeric_limits<std::uint64_t>::max()};
    constexpr std::array<std::int64_t, 8> i64_a_values{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int64_t>::max() - 1,
        std::numeric_limits<std::int64_t>::max(),
        1234567890123456789LL};
    constexpr std::array<std::int64_t, 8> i64_b_values{
        -1, -1, std::numeric_limits<std::int64_t>::min(), 0, 1, 1,
        std::numeric_limits<std::int64_t>::max(),
        -1234567890123456789LL};

    std::vector<std::int8_t> i8_a(length), i8_b(length);
    std::vector<std::uint16_t> u16_a(length), u16_b(length);
    std::vector<std::int16_t> i16_a(length), i16_b(length);
    std::vector<std::uint32_t> u32_a(length), u32_b(length);
    std::vector<std::int32_t> i32_a(length), i32_b(length);
    std::vector<std::uint64_t> u64_a(length), u64_b(length);
    std::vector<std::int64_t> i64_a(length), i64_b(length);
    for (std::size_t i = 0; i < length; ++i) {
        if (extrema) {
            i8_a[i] = i8_a_values[i & 7];
            i8_b[i] = i8_b_values[i & 7];
            u16_a[i] = u16_a_values[i & 7];
            u16_b[i] = u16_b_values[i & 7];
            i16_a[i] = i16_a_values[i & 7];
            i16_b[i] = i16_b_values[i & 7];
            u32_a[i] = u32_a_values[i & 7];
            u32_b[i] = u32_b_values[i & 7];
            i32_a[i] = i32_a_values[i & 7];
            i32_b[i] = i32_b_values[i & 7];
            u64_a[i] = u64_a_values[i & 7];
            u64_b[i] = u64_b_values[i & 7];
            i64_a[i] = i64_a_values[i & 7];
            i64_b[i] = i64_b_values[i & 7];
        } else {
            const auto i8_random = [&]() {
                const auto magnitude =
                    static_cast<std::int8_t>(rng.next() & 0x7fU);
                return (rng.next() & 1U) == 0 ? magnitude
                                              : static_cast<std::int8_t>(
                                                    -magnitude);
            };
            const auto i16_random = [&]() {
                const auto magnitude =
                    static_cast<std::int16_t>(rng.next() & 0x7fffU);
                return (rng.next() & 1U) == 0 ? magnitude
                                              : static_cast<std::int16_t>(
                                                    -magnitude);
            };
            const auto i32_random = [&]() {
                const auto magnitude =
                    static_cast<std::int32_t>(rng.next() & 0x7fffffffU);
                return (rng.next() & 1U) == 0 ? magnitude
                                              : static_cast<std::int32_t>(
                                                    -magnitude);
            };
            const auto i64_random = [&]() {
                const auto magnitude = static_cast<std::int64_t>(
                    rng.next() & 0x7fffffffffffffffULL);
                return (rng.next() & 1U) == 0
                           ? magnitude
                           : static_cast<std::int64_t>(-magnitude);
            };
            i8_a[i] = i8_random();
            i8_b[i] = i8_random();
            u16_a[i] = static_cast<std::uint16_t>(rng.next());
            u16_b[i] = static_cast<std::uint16_t>(rng.next());
            i16_a[i] = i16_random();
            i16_b[i] = i16_random();
            u32_a[i] = static_cast<std::uint32_t>(rng.next());
            u32_b[i] = static_cast<std::uint32_t>(rng.next());
            i32_a[i] = i32_random();
            i32_b[i] = i32_random();
            u64_a[i] = rng.next();
            u64_b[i] = rng.next();
            i64_a[i] = i64_random();
            i64_b[i] = i64_random();
        }
    }

    std::vector<std::int8_t> i8_expected(length), i8_actual(length);
    std::vector<std::uint16_t> u16_expected(length), u16_actual(length);
    std::vector<std::int16_t> i16_expected(length), i16_actual(length);
    std::vector<std::uint32_t> u32_expected(length), u32_actual(length);
    std::vector<std::int32_t> i32_expected(length), i32_actual(length);
    std::vector<std::uint64_t> u64_expected(length), u64_actual(length);
    std::vector<std::int64_t> i64_expected(length), i64_actual(length);
    for (std::size_t i = 0; i < length; ++i) {
        i8_expected[i] = sat_add_reference(i8_a[i], i8_b[i]);
        u16_expected[i] = sat_add_reference(u16_a[i], u16_b[i]);
        i16_expected[i] = sat_add_reference(i16_a[i], i16_b[i]);
        u32_expected[i] = sat_add_reference(u32_a[i], u32_b[i]);
        i32_expected[i] = sat_add_reference(i32_a[i], i32_b[i]);
        u64_expected[i] = sat_add_reference(u64_a[i], u64_b[i]);
        i64_expected[i] = sat_add_reference(i64_a[i], i64_b[i]);
    }

    simd_lab::sat_add_i8_scalar(i8_actual, i8_a, i8_b);
    simd_lab::sat_add_u16_scalar(u16_actual, u16_a, u16_b);
    simd_lab::sat_add_i16_scalar(i16_actual, i16_a, i16_b);
    simd_lab::sat_add_u32_scalar(u32_actual, u32_a, u32_b);
    simd_lab::sat_add_i32_scalar(i32_actual, i32_a, i32_b);
    simd_lab::sat_add_u64_scalar(u64_actual, u64_a, u64_b);
    simd_lab::sat_add_i64_scalar(i64_actual, i64_a, i64_b);
    const auto check = [length](const auto& actual, const auto& expected,
                                const char* type) {
        if (actual != expected) {
            std::cerr << "saturating-add " << type << " mismatch at length "
                      << length << '\n';
            return false;
        }
        return true;
    };
    return check(i8_actual, i8_expected, "i8") &&
           check(u16_actual, u16_expected, "u16") &&
           check(i16_actual, i16_expected, "i16") &&
           check(u32_actual, u32_expected, "u32") &&
           check(i32_actual, i32_expected, "i32") &&
           check(u64_actual, u64_expected, "u64") &&
           check(i64_actual, i64_expected, "i64");
}

bool run_saturating_sub_case(std::size_t length, XorShift64& rng,
                             bool extrema) {
    constexpr std::array<std::uint8_t, 8> u8_a_values{
        0, 1, 127, 128, 254, 255, 3, 200};
    constexpr std::array<std::uint8_t, 8> u8_b_values{
        1, 255, 0, 255, 1, 0, 200, 3};
    constexpr std::array<std::int8_t, 8> i8_a_values{
        std::numeric_limits<std::int8_t>::min(),
        std::numeric_limits<std::int8_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int8_t>::max() - 1,
        std::numeric_limits<std::int8_t>::max(), 42};
    constexpr std::array<std::int8_t, 8> i8_b_values{
        1, -1, std::numeric_limits<std::int8_t>::min(), 0, 1,
        std::numeric_limits<std::int8_t>::max(), -1, -42};
    constexpr std::array<std::uint16_t, 8> u16_a_values{
        0, 1, std::numeric_limits<std::uint16_t>::max() / 2,
        std::numeric_limits<std::uint16_t>::max() - 1,
        std::numeric_limits<std::uint16_t>::max(), 0x8000, 17, 0xffff};
    constexpr std::array<std::uint16_t, 8> u16_b_values{
        std::numeric_limits<std::uint16_t>::max(), 0,
        std::numeric_limits<std::uint16_t>::max(),
        std::numeric_limits<std::uint16_t>::max(), 1, 0xffff, 18, 0};
    constexpr std::array<std::int16_t, 8> i16_a_values{
        std::numeric_limits<std::int16_t>::min(),
        std::numeric_limits<std::int16_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int16_t>::max() - 1,
        std::numeric_limits<std::int16_t>::max(), 12345};
    constexpr std::array<std::int16_t, 8> i16_b_values{
        1, -1, std::numeric_limits<std::int16_t>::min(), 0, 1,
        std::numeric_limits<std::int16_t>::max(), -1, -12345};
    constexpr std::array<std::uint32_t, 8> u32_a_values{
        0U, 1U, std::numeric_limits<std::uint32_t>::max() / 2,
        std::numeric_limits<std::uint32_t>::max() - 1,
        std::numeric_limits<std::uint32_t>::max(), 0x80000000U, 17U,
        0x01234567U};
    constexpr std::array<std::uint32_t, 8> u32_b_values{
        std::numeric_limits<std::uint32_t>::max(), 0U,
        std::numeric_limits<std::uint32_t>::max(),
        std::numeric_limits<std::uint32_t>::max(), 1U, 0xffffffffU, 18U,
        0U};
    constexpr std::array<std::int32_t, 8> i32_a_values{
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int32_t>::max() - 1,
        std::numeric_limits<std::int32_t>::max(), 123456789};
    constexpr std::array<std::int32_t, 8> i32_b_values{
        1, -1, std::numeric_limits<std::int32_t>::min(), 0, 1,
        std::numeric_limits<std::int32_t>::max(), -1, -123456789};
    constexpr std::array<std::uint64_t, 8> u64_a_values{
        0, 1, std::numeric_limits<std::uint64_t>::max() / 2,
        std::numeric_limits<std::uint64_t>::max() - 1,
        std::numeric_limits<std::uint64_t>::max(), std::uint64_t{1} << 63,
        17, 0x0123456789abcdefULL};
    constexpr std::array<std::uint64_t, 8> u64_b_values{
        std::numeric_limits<std::uint64_t>::max(), 0,
        std::numeric_limits<std::uint64_t>::max(),
        std::numeric_limits<std::uint64_t>::max(), 1,
        std::numeric_limits<std::uint64_t>::max(), 18, 0};
    constexpr std::array<std::int64_t, 8> i64_a_values{
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::min() + 1, -1, 0, 1,
        std::numeric_limits<std::int64_t>::max() - 1,
        std::numeric_limits<std::int64_t>::max(), 1234567890123456789LL};
    constexpr std::array<std::int64_t, 8> i64_b_values{
        1, -1, std::numeric_limits<std::int64_t>::min(), 0, 1,
        std::numeric_limits<std::int64_t>::max(), -1,
        -1234567890123456789LL};

    std::vector<std::uint8_t> u8_a(length), u8_b(length);
    std::vector<std::int8_t> i8_a(length), i8_b(length);
    std::vector<std::uint16_t> u16_a(length), u16_b(length);
    std::vector<std::int16_t> i16_a(length), i16_b(length);
    std::vector<std::uint32_t> u32_a(length), u32_b(length);
    std::vector<std::int32_t> i32_a(length), i32_b(length);
    std::vector<std::uint64_t> u64_a(length), u64_b(length);
    std::vector<std::int64_t> i64_a(length), i64_b(length);

    const auto random_i8 = [&]() {
        const auto magnitude =
            static_cast<std::int8_t>(rng.next() & 0x7fU);
        return (rng.next() & 1U) == 0
                   ? magnitude
                   : static_cast<std::int8_t>(-magnitude);
    };
    const auto random_i16 = [&]() {
        const auto magnitude =
            static_cast<std::int16_t>(rng.next() & 0x7fffU);
        return (rng.next() & 1U) == 0
                   ? magnitude
                   : static_cast<std::int16_t>(-magnitude);
    };
    const auto random_i32 = [&]() {
        const auto magnitude =
            static_cast<std::int32_t>(rng.next() & 0x7fffffffU);
        return (rng.next() & 1U) == 0
                   ? magnitude
                   : static_cast<std::int32_t>(-magnitude);
    };
    const auto random_i64 = [&]() {
        const auto magnitude = static_cast<std::int64_t>(
            rng.next() & 0x7fffffffffffffffULL);
        return (rng.next() & 1U) == 0
                   ? magnitude
                   : static_cast<std::int64_t>(-magnitude);
    };
    for (std::size_t i = 0; i < length; ++i) {
        if (extrema) {
            u8_a[i] = u8_a_values[i & 7U];
            u8_b[i] = u8_b_values[i & 7U];
            i8_a[i] = i8_a_values[i & 7U];
            i8_b[i] = i8_b_values[i & 7U];
            u16_a[i] = u16_a_values[i & 7U];
            u16_b[i] = u16_b_values[i & 7U];
            i16_a[i] = i16_a_values[i & 7U];
            i16_b[i] = i16_b_values[i & 7U];
            u32_a[i] = u32_a_values[i & 7U];
            u32_b[i] = u32_b_values[i & 7U];
            i32_a[i] = i32_a_values[i & 7U];
            i32_b[i] = i32_b_values[i & 7U];
            u64_a[i] = u64_a_values[i & 7U];
            u64_b[i] = u64_b_values[i & 7U];
            i64_a[i] = i64_a_values[i & 7U];
            i64_b[i] = i64_b_values[i & 7U];
        } else {
            u8_a[i] = static_cast<std::uint8_t>(rng.next());
            u8_b[i] = static_cast<std::uint8_t>(rng.next());
            i8_a[i] = random_i8();
            i8_b[i] = random_i8();
            u16_a[i] = static_cast<std::uint16_t>(rng.next());
            u16_b[i] = static_cast<std::uint16_t>(rng.next());
            i16_a[i] = random_i16();
            i16_b[i] = random_i16();
            u32_a[i] = static_cast<std::uint32_t>(rng.next());
            u32_b[i] = static_cast<std::uint32_t>(rng.next());
            i32_a[i] = random_i32();
            i32_b[i] = random_i32();
            u64_a[i] = rng.next();
            u64_b[i] = rng.next();
            i64_a[i] = random_i64();
            i64_b[i] = random_i64();
        }
    }

    std::vector<std::uint8_t> u8_expected(length), u8_actual(length);
    std::vector<std::int8_t> i8_expected(length), i8_actual(length);
    std::vector<std::uint16_t> u16_expected(length), u16_actual(length);
    std::vector<std::int16_t> i16_expected(length), i16_actual(length);
    std::vector<std::uint32_t> u32_expected(length), u32_actual(length);
    std::vector<std::int32_t> i32_expected(length), i32_actual(length);
    std::vector<std::uint64_t> u64_expected(length), u64_actual(length);
    std::vector<std::int64_t> i64_expected(length), i64_actual(length);
    for (std::size_t i = 0; i < length; ++i) {
        u8_expected[i] = sat_sub_reference(u8_a[i], u8_b[i]);
        i8_expected[i] = sat_sub_reference(i8_a[i], i8_b[i]);
        u16_expected[i] = sat_sub_reference(u16_a[i], u16_b[i]);
        i16_expected[i] = sat_sub_reference(i16_a[i], i16_b[i]);
        u32_expected[i] = sat_sub_reference(u32_a[i], u32_b[i]);
        i32_expected[i] = sat_sub_reference(i32_a[i], i32_b[i]);
        u64_expected[i] = sat_sub_reference(u64_a[i], u64_b[i]);
        i64_expected[i] = sat_sub_reference(i64_a[i], i64_b[i]);
    }

    simd_lab::sat_sub_u8_scalar(u8_actual, u8_a, u8_b);
    if (u8_actual != u8_expected) {
        std::cerr << "saturating-sub u8 scalar mismatch at length "
                  << length << '\n';
        return false;
    }
    std::fill(u8_actual.begin(), u8_actual.end(), std::uint8_t{0});
    simd_lab::sat_sub_u8_best(u8_actual, u8_a, u8_b);
    const auto check = [length](const auto& actual, const auto& expected,
                                const char* type) {
        if (actual != expected) {
            std::cerr << "saturating-sub " << type << " mismatch at length "
                      << length << '\n';
            return false;
        }
        return true;
    };
    simd_lab::sat_sub_i8_scalar(i8_actual, i8_a, i8_b);
    simd_lab::sat_sub_u16_scalar(u16_actual, u16_a, u16_b);
    simd_lab::sat_sub_i16_scalar(i16_actual, i16_a, i16_b);
    simd_lab::sat_sub_u32_scalar(u32_actual, u32_a, u32_b);
    simd_lab::sat_sub_i32_scalar(i32_actual, i32_a, i32_b);
    simd_lab::sat_sub_u64_scalar(u64_actual, u64_a, u64_b);
    simd_lab::sat_sub_i64_scalar(i64_actual, i64_a, i64_b);
    return check(u8_actual, u8_expected, "u8-best") &&
           check(i8_actual, i8_expected, "i8") &&
           check(u16_actual, u16_expected, "u16") &&
           check(i16_actual, i16_expected, "i16") &&
           check(u32_actual, u32_expected, "u32") &&
           check(i32_actual, i32_expected, "i32") &&
           check(u64_actual, u64_expected, "u64") &&
           check(i64_actual, i64_expected, "i64");
}


std::uint8_t blend_u8_reference(std::uint8_t a, std::uint8_t b,
                                std::uint16_t weight) noexcept {
    const auto weighted_b = static_cast<std::uint32_t>(weight);
    const auto weighted_a = 256U - weighted_b;
    const auto sum =
        static_cast<std::uint32_t>(a) * weighted_a +
        static_cast<std::uint32_t>(b) * weighted_b + 128U;
    return static_cast<std::uint8_t>(sum >> 8U);
}

std::uint8_t convolve3_u8_reference(const std::vector<std::uint8_t>& src,
                                    std::size_t i) noexcept {
    const auto left = i == 0U ? 0U : i - 1U;
    const auto right = i + 1U < src.size() ? i + 1U : src.size() - 1U;
    const auto sum =
        static_cast<std::uint32_t>(src[left]) +
        2U * static_cast<std::uint32_t>(src[i]) +
        static_cast<std::uint32_t>(src[right]) + 2U;
    return static_cast<std::uint8_t>(sum >> 2U);
}

std::uint8_t convolve5_u8_reference(const std::vector<std::uint8_t>& src,
                                    std::size_t i) noexcept {
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

bool run_image_kernel_case(std::size_t length, XorShift64& rng,
                           bool extrema) {
    constexpr std::array<std::uint8_t, 8> values{
        0, 1, 2, 127, 128, 253, 254, 255};
    constexpr std::array<std::uint16_t, 7> weights{
        0, 1, 77, 128, 255, 256, 173};
    std::vector<std::uint8_t> a(length), b(length), src(length);
    for (std::size_t i = 0; i < length; ++i) {
        if (extrema) {
            a[i] = values[i & 7U];
            b[i] = values[(i + 3U) & 7U];
            src[i] = values[(i + 5U) & 7U];
        } else {
            a[i] = static_cast<std::uint8_t>(rng.next());
            b[i] = static_cast<std::uint8_t>(rng.next());
            src[i] = static_cast<std::uint8_t>(rng.next());
        }
    }

    std::vector<std::uint8_t> expected(length), actual(length);
    for (const auto weight : weights) {
        for (std::size_t i = 0; i < length; ++i) {
            expected[i] = blend_u8_reference(a[i], b[i], weight);
        }
        std::fill(actual.begin(), actual.end(), std::uint8_t{0xa5});
        simd_lab::blend_u8_scalar(actual, a, b, weight);
        if (actual != expected) {
            std::cerr << "u8 blend mismatch at length " << length
                      << ", weight " << weight << '\n';
            return false;
        }
    }

    for (std::size_t i = 0; i < length; ++i) {
        expected[i] = convolve3_u8_reference(src, i);
    }
    std::fill(actual.begin(), actual.end(), std::uint8_t{0xa5});
    simd_lab::convolve3_u8_scalar(actual, src);
    if (actual != expected) {
        std::cerr << "u8 3-tap convolution mismatch at length " << length
                  << '\n';
        return false;
    }

    for (std::size_t i = 0; i < length; ++i) {
        expected[i] = convolve5_u8_reference(src, i);
    }
    std::fill(actual.begin(), actual.end(), std::uint8_t{0xa5});
    simd_lab::convolve5_u8_scalar(actual, src);
    if (actual != expected) {
        std::cerr << "u8 5-tap convolution mismatch at length " << length
                  << '\n';
        return false;
    }
    return true;
}

bool run_image_kernel_tests() {
    constexpr std::array<std::size_t, 24> pathological_lengths{
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
        31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257};
    XorShift64 rng{0x6d2b79f5a4c38127ULL};
    for (const auto length : pathological_lengths) {
        if (!run_image_kernel_case(length, rng, true)) return false;
    }
    for (std::size_t trial = 0; trial < 256; ++trial) {
        const auto length = trial < 96U ? trial : rng.next() % 2050U;
        if (!run_image_kernel_case(length, rng, false)) return false;
    }
    return true;
}



bool run_randomized_differential_tests() {
    constexpr std::array<std::size_t, 20> pathological_lengths{
        0, 1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 63,
        64, 65, 127, 255, 256, 257};
    constexpr std::array<std::uint16_t, 8> half_values{
        0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400};
    XorShift64 rng{0x8f3ca516d27b49e1ULL};
    for (const auto length : pathological_lengths) {
        if (!run_new_operation_case(length, rng, true) ||
            !run_saturating_add_case(length, rng, true) ||
            !run_saturating_sub_case(length, rng, true)) {
            return false;
        }
    }


    for (std::size_t trial = 0; trial < 256; ++trial) {
        const std::size_t length = trial < 64 ? trial : rng.next() % 2050;
        if (!run_new_operation_case(length, rng, false) ||
            !run_saturating_add_case(length, rng, false) ||
            !run_saturating_sub_case(length, rng, false)) {
            return false;
        }
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

bool run_exhaustive_saturating_sub_test() {
    constexpr std::size_t pair_count = 256 * 256;
    std::vector<std::uint8_t> a(pair_count), b(pair_count),
        expected(pair_count), candidate(pair_count);

    for (std::size_t x = 0; x < 256; ++x) {
        for (std::size_t y = 0; y < 256; ++y) {
            const auto index = x * 256 + y;
            a[index] = static_cast<std::uint8_t>(x);
            b[index] = static_cast<std::uint8_t>(y);
            expected[index] = static_cast<std::uint8_t>(
                x < y ? 0U : x - y);
        }
    }

    simd_lab::sat_sub_u8_scalar(candidate, a, b);
    if (candidate != expected) {
        std::cerr << "scalar saturating-sub exhaustive test failed\n";
        return false;
    }
    std::fill(candidate.begin(), candidate.end(), std::uint8_t{0});
    simd_lab::sat_sub_u8_best(candidate, a, b);
    if (candidate != expected) {
        std::cerr << "dispatched saturating-sub exhaustive test failed\n";
        return false;
    }
    return true;
}


std::uint16_t f32_to_u16_sat_reference(float value) noexcept {
    constexpr float max_value =
        static_cast<float>(std::numeric_limits<std::uint16_t>::max());
    if (!(value > 0.0F)) {
        return 0;
    }
    if (value >= max_value) {
        return std::numeric_limits<std::uint16_t>::max();
    }
    return static_cast<std::uint16_t>(value);
}

std::uint8_t f32_to_u8_sat_reference(float value) noexcept {
    if (!(value > 0.0F)) {
        return 0;
    }
    if (value >= 255.0F) {
        return std::numeric_limits<std::uint8_t>::max();
    }
    return static_cast<std::uint8_t>(value);
}

bool run_mixed_width_case(std::size_t length, XorShift64& rng,
                          bool pathological) {
    constexpr std::array<std::uint8_t, 8> u8_values{
        0, 1, 2, 127, 128, 254, 255, 0xa5};
    constexpr std::array<std::int8_t, 8> i8_values{
        std::numeric_limits<std::int8_t>::min(), -127, -1, 0,
        1, 126, std::numeric_limits<std::int8_t>::max(), -42};
    constexpr std::array<std::int16_t, 8> i16_values{
        std::numeric_limits<std::int16_t>::min(),
        std::numeric_limits<std::int16_t>::min() + 1, -1, 0,
        1, std::numeric_limits<std::int16_t>::max() - 1,
        std::numeric_limits<std::int16_t>::max(), -12345};
    constexpr std::array<std::uint16_t, 8> u16_values{
        0, 1, 255, 256, 0x7fff, 0x8000, 0xfffe, 0xffff};
    const auto random_i8 = [&]() noexcept {
        const auto raw = static_cast<int>(rng.next() & 0xffU);
        return static_cast<std::int8_t>(raw < 128 ? raw : raw - 256);
    };
    const auto random_i16 = [&]() noexcept {
        const auto raw = static_cast<std::int32_t>(rng.next() & 0xffffU);
        return static_cast<std::int16_t>(raw < 32768 ? raw : raw - 65536);
    };

    std::vector<std::uint8_t> u8(length);
    std::vector<std::int8_t> i8(length);
    std::vector<std::int16_t> i16(length);
    std::vector<std::uint16_t> u16(length);
    for (std::size_t i = 0; i < length; ++i) {
        if (pathological) {
            u8[i] = u8_values[i & 7U];
            i8[i] = i8_values[i & 7U];
            i16[i] = i16_values[i & 7U];
            u16[i] = u16_values[i & 7U];
        } else {
            u8[i] = static_cast<std::uint8_t>(rng.next());
            i8[i] = random_i8();
            i16[i] = random_i16();
            u16[i] = static_cast<std::uint16_t>(rng.next());
        }
    }

    std::vector<std::uint16_t> u8_to_u16(length);
    std::vector<std::uint32_t> u8_to_u32(length), u16_to_u32(length);
    std::vector<std::int16_t> i8_to_i16(length);
    std::vector<std::int32_t> i16_to_i32(length);
    simd_lab::widen_u8_to_u16_scalar(u8_to_u16, u8);
    simd_lab::widen_u8_to_u32_scalar(u8_to_u32, u8);
    simd_lab::widen_i8_to_i16_scalar(i8_to_i16, i8);
    simd_lab::widen_i16_to_i32_scalar(i16_to_i32, i16);
    simd_lab::widen_u16_to_u32_scalar(u16_to_u32, u16);
    for (std::size_t i = 0; i < length; ++i) {
        if (u8_to_u16[i] != static_cast<std::uint16_t>(u8[i]) ||
            u8_to_u32[i] != static_cast<std::uint32_t>(u8[i]) ||
            i8_to_i16[i] != static_cast<std::int16_t>(i8[i]) ||
            i16_to_i32[i] != static_cast<std::int32_t>(i16[i]) ||
            u16_to_u32[i] != static_cast<std::uint32_t>(u16[i])) {
            std::cerr << "mixed widening mismatch at length " << length
                      << '\n';
            return false;
        }
    }

    std::vector<float> affine(length), u16_as_f32(length), i16_as_f32(length);
    const float scale = pathological ? -1.25F : 0.75F;
    const float bias = pathological ? 3.5F : -2.25F;
    simd_lab::convert_u8_f32_affine_scalar(affine, u8, scale, bias);
    simd_lab::convert_u16_to_f32_scalar(u16_as_f32, u16);
    simd_lab::convert_i16_to_f32_scalar(i16_as_f32, i16);
    for (std::size_t i = 0; i < length; ++i) {
        const float expected_affine =
            static_cast<float>(u8[i]) * scale + bias;
        if (affine[i] != expected_affine ||
            u16_as_f32[i] != static_cast<float>(u16[i]) ||
            i16_as_f32[i] != static_cast<float>(i16[i])) {
            std::cerr << "mixed conversion mismatch at length " << length
                      << '\n';
            return false;
        }
    }

    const std::array<float, 12> saturation_values{
        -std::numeric_limits<float>::infinity(), -1.0F, -0.0F, 0.0F,
        0.5F, 1.5F, 254.999F, 255.0F, 65534.9F, 65535.0F,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()};
    std::vector<float> f32_values(length);
    for (std::size_t i = 0; i < length; ++i) {
        f32_values[i] = pathological
                            ? saturation_values[i % saturation_values.size()]
                            : rng.next_float();
    }
    std::vector<std::uint16_t> u16_actual(length), u16_expected(length);
    simd_lab::f32_to_u16_sat_scalar(u16_actual, f32_values);
    for (std::size_t i = 0; i < length; ++i) {
        u16_expected[i] = f32_to_u16_sat_reference(f32_values[i]);
    }
    if (u16_actual != u16_expected) {
        std::cerr << "f32/u16 saturation mismatch at length " << length
                  << '\n';
        return false;
    }

    constexpr std::array<float, 10> valid_u8_values{
        0.0F, 0.49F, 0.5F, 0.99F, 1.0F,
        1.5F, 127.49F, 127.5F, 254.5F, 255.0F};
    std::vector<float> valid_f32(length);
    for (std::size_t i = 0; i < length; ++i) {
        valid_f32[i] = pathological
                           ? valid_u8_values[i % valid_u8_values.size()]
                           : static_cast<float>(rng.next() % 25501U) / 100.0F;
    }
    std::vector<std::uint8_t> trunc_actual(length), round_actual(length);
    std::vector<std::uint8_t> trunc_expected(length), round_expected(length);
    simd_lab::convert_f32_u8_trunc_scalar(trunc_actual, valid_f32);
    simd_lab::convert_f32_u8_round_scalar(round_actual, valid_f32);
    for (std::size_t i = 0; i < length; ++i) {
        trunc_expected[i] = static_cast<std::uint8_t>(valid_f32[i]);
        round_expected[i] = static_cast<std::uint8_t>(
            std::floor(valid_f32[i] + 0.5F));
    }
    if (trunc_actual != trunc_expected || round_actual != round_expected) {
        std::cerr << "f32/u8 truncation or rounding mismatch at length "
                  << length << '\n';
        return false;
    }

    std::vector<float> sat_f32(length);
    for (std::size_t i = 0; i < length; ++i) {
        sat_f32[i] = pathological
                         ? saturation_values[i % saturation_values.size()]
                         : rng.next_float();
    }
    std::vector<std::uint8_t> sat_u8_actual(length), sat_u8_expected(length);
    simd_lab::convert_f32_u8_sat_scalar(sat_u8_actual, sat_f32);
    for (std::size_t i = 0; i < length; ++i) {
        sat_u8_expected[i] = f32_to_u8_sat_reference(sat_f32[i]);
    }
    if (sat_u8_actual != sat_u8_expected) {
        std::cerr << "f32/u8 saturation mismatch at length " << length
                  << '\n';
        return false;
    }

    std::vector<std::uint8_t> narrow_trunc(length), narrow_round(length),
        narrow_sat(length);
    std::vector<std::uint8_t> narrow_trunc_expected(length),
        narrow_round_expected(length), narrow_sat_expected(length);
    simd_lab::narrow_u16_to_u8_trunc_scalar(narrow_trunc, u16);
    simd_lab::narrow_u16_to_u8_round_scalar(narrow_round, u16);
    simd_lab::narrow_u16_to_u8_sat_scalar(narrow_sat, u16);
    for (std::size_t i = 0; i < length; ++i) {
        narrow_trunc_expected[i] =
            static_cast<std::uint8_t>(u16[i] & 0xffU);
        narrow_round_expected[i] = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(u16[i]) + 128U) / 257U);
        narrow_sat_expected[i] = static_cast<std::uint8_t>(
            std::min<std::uint16_t>(u16[i], 255U));
    }
    if (narrow_trunc != narrow_trunc_expected ||
        narrow_round != narrow_round_expected ||
        narrow_sat != narrow_sat_expected) {
        std::cerr << "u16/u8 narrowing mismatch at length " << length
                  << '\n';
        return false;
    }

    const std::size_t groups = length;
    std::vector<std::uint8_t> bytes(groups * 4U), unpacked(groups * 4U);
    std::vector<std::uint32_t> packed(groups), packed_expected(groups);
    for (std::size_t i = 0; i < groups; ++i) {
        const std::size_t base = i * 4U;
        bytes[base] = static_cast<std::uint8_t>(
            pathological && i == 0 ? 0x01U : rng.next());
        bytes[base + 1U] = static_cast<std::uint8_t>(
            pathological && i == 0 ? 0x23U : rng.next());
        bytes[base + 2U] = static_cast<std::uint8_t>(
            pathological && i == 0 ? 0x45U : rng.next());
        bytes[base + 3U] = static_cast<std::uint8_t>(
            pathological && i == 0 ? 0x67U : rng.next());
        packed_expected[i] =
            static_cast<std::uint32_t>(bytes[base]) |
            (static_cast<std::uint32_t>(bytes[base + 1U]) << 8U) |
            (static_cast<std::uint32_t>(bytes[base + 2U]) << 16U) |
            (static_cast<std::uint32_t>(bytes[base + 3U]) << 24U);
    }
    simd_lab::pack_u8x4_to_u32_scalar(packed, bytes);
    if (packed != packed_expected ||
        (pathological && groups != 0U && packed[0] != 0x67452301U)) {
        std::cerr << "u8x4/u32 packing mismatch at group count " << groups
                  << '\n';
        return false;
    }
    simd_lab::unpack_u32_to_u8x4_scalar(unpacked, packed);
    if (unpacked != bytes) {
        std::cerr << "u32/u8x4 unpacking mismatch at group count " << groups
                  << '\n';
        return false;
    }
    return true;
}

bool run_mixed_width_tests() {
    constexpr std::array<std::size_t, 19> pathological_lengths{
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17,
        31, 32, 33, 63, 64, 65, 127, 128, 129};
    XorShift64 rng{0x2c6f4a1b9d37e805ULL};
    for (const auto length : pathological_lengths) {
        if (!run_mixed_width_case(length, rng, true)) {
            return false;
        }
    }
    for (std::size_t trial = 0; trial < 96; ++trial) {
        const auto length = trial < 32 ? trial : rng.next() % 2049U;
        if (!run_mixed_width_case(length, rng, false)) {
            return false;
        }
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
        !run_exhaustive_saturating_add_test() ||
        !run_exhaustive_saturating_sub_test() ||
        !run_mixed_width_tests() || !run_image_kernel_tests()) {
        return 1;
    }

    std::cout << "C++23 SIMD lab smoke test\n"
              << "Dispatch tier: " << simd_lab::dispatch_tier() << '\n'
              << "Saturating-add tier: "
              << simd_lab::sat_add_u8_dispatch_tier() << '\n'
              << "Saturating-sub tier: "
              << simd_lab::sat_sub_u8_dispatch_tier() << '\n'
              << "Image kernel checks: passed\n"
              << "AXPY checksum: " << checksum << '\n'
              << "Squared error scalar: " << scalar << '\n'
              << "Squared error best:   " << best << '\n';
}
