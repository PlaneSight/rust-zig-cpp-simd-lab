#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace simd_lab {

void axpy_scalar(std::span<float> dst, std::span<const float> x,
                 std::span<const float> y, float a);

double squared_error_scalar(std::span<const float> a,
                            std::span<const float> b);

double squared_error_best(std::span<const float> a,
                          std::span<const float> b);

std::uint64_t sad_u8_scalar(std::span<const std::uint8_t> a,
                            std::span<const std::uint8_t> b);

std::uint64_t sad_u8_best(std::span<const std::uint8_t> a,
                          std::span<const std::uint8_t> b);

bool clamp_f16c(std::uint16_t* dst, const std::uint16_t* c,
                const std::uint16_t* lo, const std::uint16_t* hi,
                std::size_t n);

std::string_view dispatch_tier() noexcept;

} // namespace simd_lab
