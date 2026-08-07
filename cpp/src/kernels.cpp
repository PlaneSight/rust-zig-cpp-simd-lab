#include "kernels.hpp"

#include <limits>
#include <algorithm>
#include <cassert>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace simd_lab {

void axpy_scalar(std::span<float> dst, std::span<const float> x,
                 std::span<const float> y, float a) {
    assert(dst.size() == x.size() && x.size() == y.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = std::fma(a, x[i], y[i]);
    }
}

double squared_error_scalar(std::span<const float> a,
                            std::span<const float> b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto d = static_cast<double>(a[i] - b[i]);
        sum += d * d;
    }
    return sum;
}

std::uint64_t sad_u8_scalar(std::span<const std::uint8_t> a,
                            std::span<const std::uint8_t> b) {
    assert(a.size() == b.size());
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

std::uint64_t sad_u16_scalar(std::span<const std::uint16_t> a,
                             std::span<const std::uint16_t> b) {
    assert(a.size() == b.size());
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto x = static_cast<std::uint32_t>(a[i]);
        const auto y = static_cast<std::uint32_t>(b[i]);
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

void sat_add_u8_scalar(std::span<std::uint8_t> dst,
                       std::span<const std::uint8_t> a,
                       std::span<const std::uint8_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<unsigned>(a[i]) +
                                 static_cast<unsigned>(b[i]);
        dst[i] = static_cast<std::uint8_t>(
            std::min(widened_sum, 255U));
    }
}

void blend_u8_scalar(std::span<std::uint8_t> dst,
                     std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b,
                     std::uint16_t weight) {
    assert(dst.size() == a.size() && a.size() == b.size());
    assert(weight <= 256U);
    const auto weighted_b = static_cast<std::uint32_t>(weight);
    const auto weighted_a = 256U - weighted_b;
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto sum =
            static_cast<std::uint32_t>(a[i]) * weighted_a +
            static_cast<std::uint32_t>(b[i]) * weighted_b + 128U;
        dst[i] = static_cast<std::uint8_t>(sum >> 8U);
    }
}

void convolve3_u8_scalar(std::span<std::uint8_t> dst,
                         std::span<const std::uint8_t> src) {
    assert(dst.size() == src.size());
    const auto n = src.size();
    if (n == 0U) return;
    if (n == 1U) {
        dst[0] = src[0];
        return;
    }

    dst[0] = static_cast<std::uint8_t>(
        (3U * static_cast<std::uint32_t>(src[0]) +
         static_cast<std::uint32_t>(src[1]) + 2U) >>
        2U);
    for (std::size_t i = 1; i + 1U < n; ++i) {
        const auto sum =
            static_cast<std::uint32_t>(src[i - 1U]) +
            2U * static_cast<std::uint32_t>(src[i]) +
            static_cast<std::uint32_t>(src[i + 1U]) + 2U;
        dst[i] = static_cast<std::uint8_t>(sum >> 2U);
    }
    dst[n - 1U] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(src[n - 2U]) +
         3U * static_cast<std::uint32_t>(src[n - 1U]) + 2U) >>
        2U);
}

void convolve5_u8_scalar(std::span<std::uint8_t> dst,
                         std::span<const std::uint8_t> src) {
    assert(dst.size() == src.size());
    const auto n = src.size();
    if (n == 0U) return;
    if (n == 1U) {
        dst[0] = src[0];
        return;
    }
    if (n == 2U) {
        dst[0] = static_cast<std::uint8_t>(
            (11U * static_cast<std::uint32_t>(src[0]) +
             5U * static_cast<std::uint32_t>(src[1]) + 8U) >>
            4U);
        dst[1] = static_cast<std::uint8_t>(
            (5U * static_cast<std::uint32_t>(src[0]) +
             11U * static_cast<std::uint32_t>(src[1]) + 8U) >>
            4U);
        return;
    }
    if (n == 3U) {
        dst[0] = static_cast<std::uint8_t>(
            (11U * static_cast<std::uint32_t>(src[0]) +
             4U * static_cast<std::uint32_t>(src[1]) +
             static_cast<std::uint32_t>(src[2]) + 8U) >>
            4U);
        dst[1] = static_cast<std::uint8_t>(
            (5U * static_cast<std::uint32_t>(src[0]) +
             6U * static_cast<std::uint32_t>(src[1]) +
             5U * static_cast<std::uint32_t>(src[2]) + 8U) >>
            4U);
        dst[2] = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(src[0]) +
             4U * static_cast<std::uint32_t>(src[1]) +
             11U * static_cast<std::uint32_t>(src[2]) + 8U) >>
            4U);
        return;
    }
    if (n == 4U) {
        dst[0] = static_cast<std::uint8_t>(
            (11U * static_cast<std::uint32_t>(src[0]) +
             4U * static_cast<std::uint32_t>(src[1]) +
             static_cast<std::uint32_t>(src[2]) + 8U) >>
            4U);
        dst[1] = static_cast<std::uint8_t>(
            (5U * static_cast<std::uint32_t>(src[0]) +
             6U * static_cast<std::uint32_t>(src[1]) +
             4U * static_cast<std::uint32_t>(src[2]) +
             static_cast<std::uint32_t>(src[3]) + 8U) >>
            4U);
        dst[2] = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(src[0]) +
             4U * static_cast<std::uint32_t>(src[1]) +
             6U * static_cast<std::uint32_t>(src[2]) +
             5U * static_cast<std::uint32_t>(src[3]) + 8U) >>
            4U);
        dst[3] = static_cast<std::uint8_t>(
            (static_cast<std::uint32_t>(src[1]) +
             4U * static_cast<std::uint32_t>(src[2]) +
             11U * static_cast<std::uint32_t>(src[3]) + 8U) >>
            4U);
        return;
    }

    dst[0] = static_cast<std::uint8_t>(
        (11U * static_cast<std::uint32_t>(src[0]) +
         4U * static_cast<std::uint32_t>(src[1]) +
         static_cast<std::uint32_t>(src[2]) + 8U) >>
        4U);
    dst[1] = static_cast<std::uint8_t>(
        (5U * static_cast<std::uint32_t>(src[0]) +
         6U * static_cast<std::uint32_t>(src[1]) +
         4U * static_cast<std::uint32_t>(src[2]) +
         static_cast<std::uint32_t>(src[3]) + 8U) >>
        4U);
    for (std::size_t i = 2; i + 2U < n; ++i) {
        const auto sum =
            static_cast<std::uint32_t>(src[i - 2U]) +
            4U * static_cast<std::uint32_t>(src[i - 1U]) +
            6U * static_cast<std::uint32_t>(src[i]) +
            4U * static_cast<std::uint32_t>(src[i + 1U]) +
            static_cast<std::uint32_t>(src[i + 2U]) + 8U;
        dst[i] = static_cast<std::uint8_t>(sum >> 4U);
    }
    dst[n - 2U] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(src[n - 4U]) +
         4U * static_cast<std::uint32_t>(src[n - 3U]) +
         6U * static_cast<std::uint32_t>(src[n - 2U]) +
         5U * static_cast<std::uint32_t>(src[n - 1U]) + 8U) >>
        4U);
    dst[n - 1U] = static_cast<std::uint8_t>(
        (static_cast<std::uint32_t>(src[n - 3U]) +
         4U * static_cast<std::uint32_t>(src[n - 2U]) +
         11U * static_cast<std::uint32_t>(src[n - 1U]) + 8U) >>
        4U);
}


void sat_add_i8_scalar(std::span<std::int8_t> dst,
                       std::span<const std::int8_t> a,
                       std::span<const std::int8_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto min_value = std::numeric_limits<std::int8_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int8_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<std::int32_t>(a[i]) +
                                 static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int8_t>(std::clamp(
            widened_sum, static_cast<std::int32_t>(min_value),
            static_cast<std::int32_t>(max_value)));
    }
}

void sat_add_u16_scalar(std::span<std::uint16_t> dst,
                        std::span<const std::uint16_t> a,
                        std::span<const std::uint16_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto max_value = std::numeric_limits<std::uint16_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<std::uint32_t>(a[i]) +
                                 static_cast<std::uint32_t>(b[i]);
        dst[i] = static_cast<std::uint16_t>(
            std::min(widened_sum, static_cast<std::uint32_t>(max_value)));
    }
}

void sat_add_i16_scalar(std::span<std::int16_t> dst,
                        std::span<const std::int16_t> a,
                        std::span<const std::int16_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto min_value = std::numeric_limits<std::int16_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int16_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<std::int32_t>(a[i]) +
                                 static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int16_t>(std::clamp(
            widened_sum, static_cast<std::int32_t>(min_value),
            static_cast<std::int32_t>(max_value)));
    }
}

void sat_add_u32_scalar(std::span<std::uint32_t> dst,
                        std::span<const std::uint32_t> a,
                        std::span<const std::uint32_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto max_value = std::numeric_limits<std::uint32_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<std::uint64_t>(a[i]) +
                                 static_cast<std::uint64_t>(b[i]);
        dst[i] = static_cast<std::uint32_t>(
            std::min(widened_sum, static_cast<std::uint64_t>(max_value)));
    }
}

void sat_add_i32_scalar(std::span<std::int32_t> dst,
                        std::span<const std::int32_t> a,
                        std::span<const std::int32_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto min_value = std::numeric_limits<std::int32_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int32_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<std::int64_t>(a[i]) +
                                 static_cast<std::int64_t>(b[i]);
        dst[i] = static_cast<std::int32_t>(std::clamp(
            widened_sum, static_cast<std::int64_t>(min_value),
            static_cast<std::int64_t>(max_value)));
    }
}

void sat_add_u64_scalar(std::span<std::uint64_t> dst,
                        std::span<const std::uint64_t> a,
                        std::span<const std::uint64_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto max_value = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = a[i];
        const auto rhs = b[i];
        dst[i] = lhs > max_value - rhs ? max_value : lhs + rhs;
    }
}

void sat_add_i64_scalar(std::span<std::int64_t> dst,
                        std::span<const std::int64_t> a,
                        std::span<const std::int64_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    constexpr auto min_value = std::numeric_limits<std::int64_t>::min();
    constexpr auto max_value = std::numeric_limits<std::int64_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = a[i];
        const auto rhs = b[i];
        if (rhs > 0 && lhs > max_value - rhs) {
            dst[i] = max_value;
        } else if (rhs < 0 && lhs < min_value - rhs) {
            dst[i] = min_value;
        } else {
            dst[i] = lhs + rhs;
        }
    }
}

double dot_f32_scalar(std::span<const float> a,
                      std::span<const float> b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto lhs = static_cast<double>(a[i]);
        const auto rhs = static_cast<double>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}

double dot_f64_scalar(std::span<const double> a,
                      std::span<const double> b) {
    assert(a.size() == b.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::int64_t dot_i16_scalar(std::span<const std::int16_t> a,
                            std::span<const std::int16_t> b) {
    assert(a.size() == b.size());
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto lhs = static_cast<std::int64_t>(a[i]);
        const auto rhs = static_cast<std::int64_t>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}

std::int64_t dot_u8_i8_scalar(std::span<const std::uint8_t> a,
                              std::span<const std::int8_t> b) {
    assert(a.size() == b.size());
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto lhs = static_cast<std::int64_t>(a[i]);
        const auto rhs = static_cast<std::int64_t>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}

void widen_mul_u8_u16_scalar(std::span<std::uint16_t> dst,
                             std::span<const std::uint8_t> a,
                             std::span<const std::uint8_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::uint16_t>(a[i]);
        const auto rhs = static_cast<std::uint16_t>(b[i]);
        dst[i] = static_cast<std::uint16_t>(lhs * rhs);
    }
}

void widen_mul_i8_i16_scalar(std::span<std::int16_t> dst,
                             std::span<const std::int8_t> a,
                             std::span<const std::int8_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::int16_t>(a[i]);
        const auto rhs = static_cast<std::int16_t>(b[i]);
        dst[i] = static_cast<std::int16_t>(lhs * rhs);
    }
}

void widen_mul_u16_u32_scalar(std::span<std::uint32_t> dst,
                              std::span<const std::uint16_t> a,
                              std::span<const std::uint16_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::uint32_t>(a[i]);
        const auto rhs = static_cast<std::uint32_t>(b[i]);
        dst[i] = static_cast<std::uint32_t>(lhs * rhs);
    }
}

void widen_mul_i16_i32_scalar(std::span<std::int32_t> dst,
                              std::span<const std::int16_t> a,
                              std::span<const std::int16_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::int32_t>(a[i]);
        const auto rhs = static_cast<std::int32_t>(b[i]);
        dst[i] = static_cast<std::int32_t>(lhs * rhs);
    }
}
void widen_mul_u32_u64_scalar(std::span<std::uint64_t> dst,
                              std::span<const std::uint32_t> a,
                              std::span<const std::uint32_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::uint64_t>(a[i]);
        const auto rhs = static_cast<std::uint64_t>(b[i]);
        dst[i] = lhs * rhs;
    }
}

void widen_mul_i32_i64_scalar(std::span<std::int64_t> dst,
                              std::span<const std::int32_t> a,
                              std::span<const std::int32_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto lhs = static_cast<std::int64_t>(a[i]);
        const auto rhs = static_cast<std::int64_t>(b[i]);
        dst[i] = lhs * rhs;
    }
}

void widen_u8_to_u16_scalar(std::span<std::uint16_t> dst,
                            std::span<const std::uint8_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::uint16_t>(src[i]);
    }
}

void widen_u8_to_u32_scalar(std::span<std::uint32_t> dst,
                            std::span<const std::uint8_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::uint32_t>(src[i]);
    }
}

void widen_i8_to_i16_scalar(std::span<std::int16_t> dst,
                            std::span<const std::int8_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::int16_t>(src[i]);
    }
}

void widen_i16_to_i32_scalar(std::span<std::int32_t> dst,
                             std::span<const std::int16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::int32_t>(src[i]);
    }
}

void widen_u16_to_u32_scalar(std::span<std::uint32_t> dst,
                             std::span<const std::uint16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::uint32_t>(src[i]);
    }
}

void convert_u8_f32_affine_scalar(std::span<float> dst,
                                  std::span<const std::uint8_t> src,
                                  float scale, float bias) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<float>(src[i]) * scale + bias;
    }
}

void convert_u16_to_f32_scalar(std::span<float> dst,
                               std::span<const std::uint16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void convert_i16_to_f32_scalar(std::span<float> dst,
                               std::span<const std::int16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

void f32_to_u16_sat_scalar(std::span<std::uint16_t> dst,
                           std::span<const float> src) {
    assert(dst.size() == src.size());
    constexpr float max_value =
        static_cast<float>(std::numeric_limits<std::uint16_t>::max());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const float value = src[i];
        if (!(value > 0.0F)) {
            dst[i] = 0;
        } else if (value >= max_value) {
            dst[i] = std::numeric_limits<std::uint16_t>::max();
        } else {
            dst[i] = static_cast<std::uint16_t>(value);
        }
    }
}

void convert_f32_u8_trunc_scalar(std::span<std::uint8_t> dst,
                                 std::span<const float> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const float value = src[i];
        assert(std::isfinite(value) && value >= 0.0F && value <= 255.0F);
        if (!std::isfinite(value) || value <= 0.0F) {
            dst[i] = 0;
        } else if (value >= 255.0F) {
            dst[i] = std::numeric_limits<std::uint8_t>::max();
        } else {
            dst[i] = static_cast<std::uint8_t>(value);
        }
    }
}

void convert_f32_u8_round_scalar(std::span<std::uint8_t> dst,
                                 std::span<const float> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const float value = src[i];
        assert(std::isfinite(value) && value >= 0.0F && value <= 255.0F);
        if (!std::isfinite(value) || value <= 0.0F) {
            dst[i] = 0;
        } else if (value >= 255.0F) {
            dst[i] = std::numeric_limits<std::uint8_t>::max();
        } else {
            const float rounded = std::floor(value + 0.5F);
            dst[i] = static_cast<std::uint8_t>(rounded);
        }
    }
}

void convert_f32_u8_sat_scalar(std::span<std::uint8_t> dst,
                               std::span<const float> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const float value = src[i];
        if (!(value > 0.0F)) {
            dst[i] = 0;
        } else if (value >= 255.0F) {
            dst[i] = std::numeric_limits<std::uint8_t>::max();
        } else {
            dst[i] = static_cast<std::uint8_t>(value);
        }
    }
}

void narrow_u16_to_u8_trunc_scalar(
    std::span<std::uint8_t> dst, std::span<const std::uint16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::uint8_t>(src[i] & 0xffU);
    }
}

void narrow_u16_to_u8_round_scalar(
    std::span<std::uint8_t> dst, std::span<const std::uint16_t> src) {
    assert(dst.size() == src.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto widened = static_cast<std::uint32_t>(src[i]) + 128U;
        dst[i] = static_cast<std::uint8_t>(widened / 257U);
    }
}

void narrow_u16_to_u8_sat_scalar(
    std::span<std::uint8_t> dst, std::span<const std::uint16_t> src) {
    assert(dst.size() == src.size());
    constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();
    for (std::size_t i = 0; i < dst.size(); ++i) {
        dst[i] = static_cast<std::uint8_t>(
            std::min<std::uint16_t>(src[i], max_value));
    }
}

void pack_u8x4_to_u32_scalar(std::span<std::uint32_t> dst,
                             std::span<const std::uint8_t> src) {
    assert(src.size() % 4U == 0U);
    assert(dst.size() == src.size() / 4U);
    std::size_t src_index = 0;
    for (std::size_t i = 0; i < dst.size(); ++i) {
        const auto b0 = static_cast<std::uint32_t>(src[src_index++]);
        const auto b1 = static_cast<std::uint32_t>(src[src_index++]);
        const auto b2 = static_cast<std::uint32_t>(src[src_index++]);
        const auto b3 = static_cast<std::uint32_t>(src[src_index++]);
        dst[i] = b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
    }
}

void unpack_u32_to_u8x4_scalar(
    std::span<std::uint8_t> dst, std::span<const std::uint32_t> src) {
    assert(dst.size() % 4U == 0U);
    assert(src.size() == dst.size() / 4U);
    std::size_t dst_index = 0;
    for (const auto packed : src) {
        dst[dst_index++] = static_cast<std::uint8_t>(packed & 0xffU);
        dst[dst_index++] =
            static_cast<std::uint8_t>((packed >> 8U) & 0xffU);
        dst[dst_index++] =
            static_cast<std::uint8_t>((packed >> 16U) & 0xffU);
        dst[dst_index++] =
            static_cast<std::uint8_t>((packed >> 24U) & 0xffU);
    }
}



#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
__attribute__((target("avx2,fma")))
static double squared_error_avx2(std::span<const float> a,
                                 std::span<const float> b) {
    __m256d acc_lo = _mm256_setzero_pd();
    __m256d acc_hi = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= a.size(); i += 8) {
        const __m256 va = _mm256_loadu_ps(a.data() + i);
        const __m256 vb = _mm256_loadu_ps(b.data() + i);
        const __m256 d = _mm256_sub_ps(va, vb);
        const __m256d d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(d));
        const __m256d d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(d, 1));
        acc_lo = _mm256_fmadd_pd(d_lo, d_lo, acc_lo);
        acc_hi = _mm256_fmadd_pd(d_hi, d_hi, acc_hi);
    }

    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, _mm256_add_pd(acc_lo, acc_hi));
    double sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < a.size(); ++i) {
        const auto d = static_cast<double>(a[i] - b[i]);
        sum += d * d;
    }
    return sum;
}

__attribute__((target("avx2")))
static std::uint64_t sad_u8_avx2(std::span<const std::uint8_t> a,
                                 std::span<const std::uint8_t> b) {
    __m256i acc = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 32 <= a.size(); i += 32) {
        const __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(va, vb));
    }

    alignas(32) std::uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    std::uint64_t sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < a.size(); ++i) {
        const auto x = a[i];
        const auto y = b[i];
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

__attribute__((target("avx2")))
static std::uint64_t sad_u16_avx2(
    std::span<const std::uint16_t> a,
    std::span<const std::uint16_t> b) {
    assert(a.size() == b.size());
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    std::size_t i = 0;
    for (; i + 16 <= a.size(); i += 16) {
        const __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));
        const __m256i hi = _mm256_max_epu16(va, vb);
        const __m256i lo = _mm256_min_epu16(va, vb);
        const __m256i diff = _mm256_sub_epi16(hi, lo);

        const __m256i diff_lo =
            _mm256_cvtepu16_epi32(_mm256_castsi256_si128(diff));
        const __m256i diff_hi = _mm256_cvtepu16_epi32(
            _mm256_extracti128_si256(diff, 1));
        acc0 = _mm256_add_epi64(
            acc0, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_lo)));
        acc1 = _mm256_add_epi64(
            acc1, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff_lo, 1)));
        acc2 = _mm256_add_epi64(
            acc2, _mm256_cvtepu32_epi64(_mm256_castsi256_si128(diff_hi)));
        acc3 = _mm256_add_epi64(
            acc3, _mm256_cvtepu32_epi64(_mm256_extracti128_si256(diff_hi, 1)));
    }

    alignas(32) std::uint64_t lanes[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 4), acc1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 8), acc2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes + 12), acc3);
    std::uint64_t sum = 0;
    for (const auto lane : lanes) sum += lane;
    for (; i < a.size(); ++i) {
        const auto x = static_cast<std::uint32_t>(a[i]);
        const auto y = static_cast<std::uint32_t>(b[i]);
        sum += x > y ? static_cast<std::uint64_t>(x - y)
                     : static_cast<std::uint64_t>(y - x);
    }
    return sum;
}

__attribute__((target("avx2")))
static void sat_add_u8_avx2(std::span<std::uint8_t> dst,
                            std::span<const std::uint8_t> a,
                            std::span<const std::uint8_t> b) {
    assert(dst.size() == a.size() && a.size() == b.size());
    std::size_t i = 0;
    for (; i + 32 <= dst.size(); i += 32) {
        const __m256i va = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i vb = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));
        const __m256i result = _mm256_adds_epu8(va, vb);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(dst.data() + i), result);
    }
    for (; i < dst.size(); ++i) {
        const auto widened_sum = static_cast<unsigned>(a[i]) +
                                 static_cast<unsigned>(b[i]);
        dst[i] = static_cast<std::uint8_t>(
            std::min(widened_sum, 255U));
    }
}
#endif

#if defined(_MSC_VER) && defined(_M_X64)
double squared_error_avx2_msvc(std::span<const float> a,
                               std::span<const float> b) noexcept;
std::uint64_t sad_u8_avx2_msvc(std::span<const std::uint8_t> a,
                               std::span<const std::uint8_t> b) noexcept;
void sat_add_u8_avx2_msvc(std::span<std::uint8_t> dst,
                          std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b) noexcept;

static bool os_has_ymm_state() noexcept {
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    constexpr int osxsave = 1 << 27;
    constexpr int avx = 1 << 28;
    if ((registers[2] & (osxsave | avx)) != (osxsave | avx)) return false;
    return (_xgetbv(0) & 0x6) == 0x6;
}

static bool cpu_has_avx2() noexcept {
    if (!os_has_ymm_state()) return false;
    int registers[4]{};
    __cpuidex(registers, 7, 0);
    constexpr int avx2 = 1 << 5;
    return (registers[1] & avx2) != 0;
}

static bool cpu_has_avx2_fma() noexcept {
    if (!cpu_has_avx2()) return false;
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    constexpr int fma = 1 << 12;
    return (registers[2] & fma) != 0;
}
#endif

double squared_error_best(std::span<const float> a,
                          std::span<const float> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return squared_error_avx2(a, b);
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    if (cpu_has_avx2_fma()) return squared_error_avx2_msvc(a, b);
#endif
    return squared_error_scalar(a, b);
}

std::uint64_t sad_u8_best(std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) return sad_u8_avx2(a, b);
#elif defined(_MSC_VER) && defined(_M_X64)
    if (cpu_has_avx2()) return sad_u8_avx2_msvc(a, b);
#endif
    return sad_u8_scalar(a, b);
}

std::uint64_t sad_u16_best(std::span<const std::uint16_t> a,
                           std::span<const std::uint16_t> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) return sad_u16_avx2(a, b);
#endif
    return sad_u16_scalar(a, b);
}

void sat_add_u8_best(std::span<std::uint8_t> dst,
                     std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b) {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) {
        sat_add_u8_avx2(dst, a, b);
        return;
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    if (cpu_has_avx2()) {
        sat_add_u8_avx2_msvc(dst, a, b);
        return;
    }
#endif
    sat_add_u8_scalar(dst, a, b);
}

std::string_view dispatch_tier() noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return "avx2+fma";
    }
#elif defined(_MSC_VER) && defined(_M_X64)
    if (cpu_has_avx2_fma()) return "avx2+fma";
#endif
    return "scalar";
}

std::string_view sat_add_u8_dispatch_tier() noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
    if (__builtin_cpu_supports("avx2")) return "avx2";
#elif defined(_MSC_VER) && defined(_M_X64)
    if (cpu_has_avx2()) return "avx2";
#endif
    return "scalar";
}

} // namespace simd_lab
