#include "kernels.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    constexpr std::array<std::uint16_t, 23> cases{
        0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x37ff, 0x3800, 0x3801,
        0x3bff, 0x3c00, 0x3c01, 0x4000, 0x7bff, 0x7c00, 0x7e01, 0x7fff,
        0x8001, 0xb800, 0xbc00, 0xc000, 0xfbff, 0xfc00, 0xfe01};
    constexpr std::size_t work_len = (cases.size() + 7) / 8 * 8;
    std::array<std::uint16_t, work_len> padded_cases{}, lo{}, hi{}, out{};
    for (std::size_t i = 0; i < cases.size(); ++i) {
        padded_cases[i] = cases[i];
    }
    lo.fill(0xb800); // -0.5
    hi.fill(0x4000); // +2.0

    const bool available = simd_lab::clamp_f16c(
        out.data(), padded_cases.data(), lo.data(), hi.data(), work_len);

    std::cout << "{\"strategy\":\"cpp-f16c\",\"available\":"
              << (available ? "true" : "false") << ",\"outputs\":[";
    if (available) {
        for (std::size_t i = 0; i < cases.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '\"' << std::hex << std::setfill('0') << std::setw(4)
                      << static_cast<unsigned>(out[i]) << '\"';
        }
    }
    std::cout << "]}\n";
}
