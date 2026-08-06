// Standalone compiler/codegen probes.
//
// Examples:
//   clang++ -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o clamp-clang.s
//   g++     -std=c++23 -O3 -S -masm=intel probes/cpp/clamp.cpp -o clamp-gcc.s

#include <algorithm>
#include <cstdint>

extern "C" std::uint8_t clamp_u8(
    std::uint8_t c,
    std::uint8_t a1,
    std::uint8_t a2,
    std::uint8_t a3,
    std::uint8_t a4,
    std::uint8_t a5,
    std::uint8_t a6,
    std::uint8_t a7,
    std::uint8_t a8) {
    const auto lo = std::min({a1, a2, a3, a4, a5, a6, a7, a8});
    const auto hi = std::max({a1, a2, a3, a4, a5, a6, a7, a8});
    return std::clamp(c, lo, hi);
}

extern "C" std::uint16_t clamp_u16(
    std::uint16_t c,
    std::uint16_t a1,
    std::uint16_t a2,
    std::uint16_t a3,
    std::uint16_t a4,
    std::uint16_t a5,
    std::uint16_t a6,
    std::uint16_t a7,
    std::uint16_t a8) {
    const auto lo = std::min({a1, a2, a3, a4, a5, a6, a7, a8});
    const auto hi = std::max({a1, a2, a3, a4, a5, a6, a7, a8});
    return std::clamp(c, lo, hi);
}

extern "C" float clamp_f32(
    float c,
    float a1,
    float a2,
    float a3,
    float a4,
    float a5,
    float a6,
    float a7,
    float a8) {
    const auto lo = std::min({a1, a2, a3, a4, a5, a6, a7, a8});
    const auto hi = std::max({a1, a2, a3, a4, a5, a6, a7, a8});
    return std::clamp(c, lo, hi);
}

extern "C" double clamp_f64(
    double c,
    double a1,
    double a2,
    double a3,
    double a4,
    double a5,
    double a6,
    double a7,
    double a8) {
    const auto lo = std::min({a1, a2, a3, a4, a5, a6, a7, a8});
    const auto hi = std::max({a1, a2, a3, a4, a5, a6, a7, a8});
    return std::clamp(c, lo, hi);
}
