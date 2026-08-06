#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace simd_lab {

void axpy_scalar(std::span<float> dst, std::span<const float> x,
                 std::span<const float> y, float a);

float squared_error_scalar(std::span<const float> a,
                           std::span<const float> b);

float squared_error_best(std::span<const float> a,
                         std::span<const float> b);

bool clamp_f16c(std::uint16_t* dst, const std::uint16_t* c,
                const std::uint16_t* lo, const std::uint16_t* hi,
                std::size_t n);

} // namespace simd_lab
