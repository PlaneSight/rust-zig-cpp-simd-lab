// Architecture-neutral widening-multiply probes for cross-target autovectorization.
// Deliberately avoids headers and target-specific types so Clang can cross-compile
// without a target sysroot.

using i8 = signed char;
using i16 = short;
using i32 = int;
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using usize = __SIZE_TYPE__;

extern "C" void widen_mul_u8_u16_scalar(
    u16* dst, const u8* a, const u8* b, usize len) {
    for (usize i = 0; i < len; ++i) {
        const u16 lhs = static_cast<u16>(a[i]);
        const u16 rhs = static_cast<u16>(b[i]);
        dst[i] = static_cast<u16>(lhs * rhs);
    }
}

extern "C" void widen_mul_i8_i16_scalar(
    i16* dst, const i8* a, const i8* b, usize len) {
    for (usize i = 0; i < len; ++i) {
        const i16 lhs = static_cast<i16>(a[i]);
        const i16 rhs = static_cast<i16>(b[i]);
        dst[i] = static_cast<i16>(lhs * rhs);
    }
}

extern "C" void widen_mul_u16_u32_scalar(
    u32* dst, const u16* a, const u16* b, usize len) {
    for (usize i = 0; i < len; ++i) {
        const u32 lhs = static_cast<u32>(a[i]);
        const u32 rhs = static_cast<u32>(b[i]);
        dst[i] = static_cast<u32>(lhs * rhs);
    }
}

extern "C" void widen_mul_i16_i32_scalar(
    i32* dst, const i16* a, const i16* b, usize len) {
    for (usize i = 0; i < len; ++i) {
        const i32 lhs = static_cast<i32>(a[i]);
        const i32 rhs = static_cast<i32>(b[i]);
        dst[i] = static_cast<i32>(lhs * rhs);
    }
}
