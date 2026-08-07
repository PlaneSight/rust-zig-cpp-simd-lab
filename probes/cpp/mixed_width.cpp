// Architecture-neutral mixed-width conversion and packing probes.
// These raw-pointer entry points mirror the C++ span APIs while remaining
// suitable for standalone cross-target compiler/codegen experiments.

#if __has_include(<cmath>) && __has_include(<cstdint>) && __has_include(<limits>)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#else
namespace std {
using size_t = __SIZE_TYPE__;
using uint8_t = unsigned char;
using int8_t = signed char;
using uint16_t = unsigned short;
using int16_t = short;
using uint32_t = unsigned int;
using int32_t = int;

template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<uint8_t> {
    static constexpr uint8_t max() noexcept { return 255U; }
};

template <>
struct numeric_limits<uint16_t> {
    static constexpr uint16_t max() noexcept { return 65535U; }
};

inline bool isfinite(float value) noexcept {
    return __builtin_isfinite(value);
}

inline float floor(float value) noexcept {
    return __builtin_floorf(value);
}
} // namespace std
#endif

extern "C" void widen_u8_to_u16_scalar(std::uint16_t* dst,
                                        const std::uint8_t* src,
                                        std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::uint16_t>(src[i]);
    }
}

extern "C" void widen_u8_to_u32_scalar(std::uint32_t* dst,
                                        const std::uint8_t* src,
                                        std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::uint32_t>(src[i]);
    }
}

extern "C" void widen_i8_to_i16_scalar(std::int16_t* dst,
                                        const std::int8_t* src,
                                        std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::int16_t>(src[i]);
    }
}

extern "C" void widen_i16_to_i32_scalar(std::int32_t* dst,
                                         const std::int16_t* src,
                                         std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::int32_t>(src[i]);
    }
}

extern "C" void widen_u16_to_u32_scalar(std::uint32_t* dst,
                                         const std::uint16_t* src,
                                         std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::uint32_t>(src[i]);
    }
}

extern "C" void convert_u8_f32_affine_scalar(float* dst,
                                              const std::uint8_t* src,
                                              float scale, float bias,
                                              std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale + bias;
    }
}

extern "C" void convert_u16_to_f32_scalar(float* dst,
                                           const std::uint16_t* src,
                                           std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

extern "C" void convert_i16_to_f32_scalar(float* dst,
                                           const std::int16_t* src,
                                           std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

extern "C" void f32_to_u16_sat_scalar(std::uint16_t* dst,
                                       const float* src,
                                       std::size_t len) {
    constexpr float max_value =
        static_cast<float>(std::numeric_limits<std::uint16_t>::max());
    for (std::size_t i = 0; i < len; ++i) {
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

extern "C" void convert_f32_u8_trunc_scalar(std::uint8_t* dst,
                                             const float* src,
                                             std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const float value = src[i];
        if (!std::isfinite(value) || value <= 0.0F) {
            dst[i] = 0;
        } else if (value >= 255.0F) {
            dst[i] = std::numeric_limits<std::uint8_t>::max();
        } else {
            dst[i] = static_cast<std::uint8_t>(value);
        }
    }
}

extern "C" void convert_f32_u8_round_scalar(std::uint8_t* dst,
                                             const float* src,
                                             std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const float value = src[i];
        if (!std::isfinite(value) || value <= 0.0F) {
            dst[i] = 0;
        } else if (value >= 255.0F) {
            dst[i] = std::numeric_limits<std::uint8_t>::max();
        } else {
            dst[i] = static_cast<std::uint8_t>(std::floor(value + 0.5F));
        }
    }
}

extern "C" void convert_f32_u8_sat_scalar(std::uint8_t* dst,
                                           const float* src,
                                           std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
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

extern "C" void narrow_u16_to_u8_trunc_scalar(std::uint8_t* dst,
                                                const std::uint16_t* src,
                                                std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = static_cast<std::uint8_t>(src[i] & 0xffU);
    }
}

extern "C" void narrow_u16_to_u8_round_scalar(std::uint8_t* dst,
                                                const std::uint16_t* src,
                                                std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        const auto widened = static_cast<std::uint32_t>(src[i]) + 128U;
        dst[i] = static_cast<std::uint8_t>(widened / 257U);
    }
}

extern "C" void narrow_u16_to_u8_sat_scalar(std::uint8_t* dst,
                                              const std::uint16_t* src,
                                              std::size_t len) {
    constexpr auto max_value = std::numeric_limits<std::uint8_t>::max();
    for (std::size_t i = 0; i < len; ++i) {
        dst[i] = src[i] > max_value ? max_value
                                    : static_cast<std::uint8_t>(src[i]);
    }
}

extern "C" void pack_u8x4_to_u32_scalar(std::uint32_t* dst,
                                         const std::uint8_t* src,
                                         std::size_t groups) {
    for (std::size_t i = 0; i < groups; ++i) {
        const std::size_t base = i * 4U;
        const auto b0 = static_cast<std::uint32_t>(src[base]);
        const auto b1 = static_cast<std::uint32_t>(src[base + 1U]);
        const auto b2 = static_cast<std::uint32_t>(src[base + 2U]);
        const auto b3 = static_cast<std::uint32_t>(src[base + 3U]);
        dst[i] = b0 | (b1 << 8U) | (b2 << 16U) | (b3 << 24U);
    }
}

extern "C" void unpack_u32_to_u8x4_scalar(std::uint8_t* dst,
                                            const std::uint32_t* src,
                                            std::size_t groups) {
    for (std::size_t i = 0; i < groups; ++i) {
        const std::uint32_t packed = src[i];
        const std::size_t base = i * 4U;
        dst[base] = static_cast<std::uint8_t>(packed & 0xffU);
        dst[base + 1U] = static_cast<std::uint8_t>((packed >> 8U) & 0xffU);
        dst[base + 2U] = static_cast<std::uint8_t>((packed >> 16U) & 0xffU);
        dst[base + 3U] = static_cast<std::uint8_t>((packed >> 24U) & 0xffU);
    }
}
