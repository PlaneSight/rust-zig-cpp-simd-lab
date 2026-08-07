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

void sat_add_u8_scalar(std::span<std::uint8_t> dst,
                       std::span<const std::uint8_t> a,
                       std::span<const std::uint8_t> b);

void sat_add_i8_scalar(std::span<std::int8_t> dst,
                       std::span<const std::int8_t> a,
                       std::span<const std::int8_t> b);

void sat_add_u16_scalar(std::span<std::uint16_t> dst,
                        std::span<const std::uint16_t> a,
                        std::span<const std::uint16_t> b);

void sat_add_i16_scalar(std::span<std::int16_t> dst,
                        std::span<const std::int16_t> a,
                        std::span<const std::int16_t> b);

void sat_add_u32_scalar(std::span<std::uint32_t> dst,
                        std::span<const std::uint32_t> a,
                        std::span<const std::uint32_t> b);

void sat_add_i32_scalar(std::span<std::int32_t> dst,
                        std::span<const std::int32_t> a,
                        std::span<const std::int32_t> b);

void sat_add_u64_scalar(std::span<std::uint64_t> dst,
                        std::span<const std::uint64_t> a,
                        std::span<const std::uint64_t> b);

void sat_add_i64_scalar(std::span<std::int64_t> dst,
                        std::span<const std::int64_t> a,
                        std::span<const std::int64_t> b);

void sat_add_u8_best(std::span<std::uint8_t> dst,
                     std::span<const std::uint8_t> a,
                     std::span<const std::uint8_t> b);
double dot_f32_scalar(std::span<const float> a,
                      std::span<const float> b);

double dot_f64_scalar(std::span<const double> a,
                      std::span<const double> b);

std::int64_t dot_i16_scalar(std::span<const std::int16_t> a,
                            std::span<const std::int16_t> b);

std::int64_t dot_u8_i8_scalar(std::span<const std::uint8_t> a,
                              std::span<const std::int8_t> b);

void widen_mul_u8_u16_scalar(std::span<std::uint16_t> dst,
                             std::span<const std::uint8_t> a,
                             std::span<const std::uint8_t> b);

void widen_mul_i8_i16_scalar(std::span<std::int16_t> dst,
                             std::span<const std::int8_t> a,
                             std::span<const std::int8_t> b);

void widen_mul_u16_u32_scalar(std::span<std::uint32_t> dst,
                              std::span<const std::uint16_t> a,
                              std::span<const std::uint16_t> b);

void widen_mul_i16_i32_scalar(std::span<std::int32_t> dst,
                              std::span<const std::int16_t> a,
                              std::span<const std::int16_t> b);

void widen_mul_u32_u64_scalar(std::span<std::uint64_t> dst,
                              std::span<const std::uint32_t> a,
                              std::span<const std::uint32_t> b);

void widen_mul_i32_i64_scalar(std::span<std::int64_t> dst,
                              std::span<const std::int32_t> a,
                              std::span<const std::int32_t> b);

bool clamp_f16c(std::uint16_t* dst, const std::uint16_t* c,
                const std::uint16_t* lo, const std::uint16_t* hi,
                std::size_t n);

std::string_view dispatch_tier() noexcept;
std::string_view sat_add_u8_dispatch_tier() noexcept;

} // namespace simd_lab
