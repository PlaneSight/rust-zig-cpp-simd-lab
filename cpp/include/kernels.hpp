#pragma once

#include <span>

namespace simd_lab {

void axpy_scalar(std::span<float> dst, std::span<const float> x,
                 std::span<const float> y, float a);

float squared_error_scalar(std::span<const float> a,
                           std::span<const float> b);

float squared_error_best(std::span<const float> a,
                         std::span<const float> b);

} // namespace simd_lab
