// Architecture-neutral dot-product probes for cross-target autovectorization.
// Deliberately avoids headers and target-specific types so Clang can cross-compile
// without a target sysroot.

using f32 = float;
using f64 = double;
using i8 = signed char;
using i16 = short;
using i64 = long long;
using u8 = unsigned char;
using usize = __SIZE_TYPE__;

extern "C" f64 dot_f32_scalar(const f32* a, const f32* b, usize len) {
    f64 sum = 0.0;
    for (usize i = 0; i < len; ++i) {
        const f64 lhs = static_cast<f64>(a[i]);
        const f64 rhs = static_cast<f64>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}

extern "C" f64 dot_f64_scalar(const f64* a, const f64* b, usize len) {
    f64 sum = 0.0;
    for (usize i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

extern "C" i64 dot_i16_scalar(const i16* a, const i16* b, usize len) {
    i64 sum = 0;
    for (usize i = 0; i < len; ++i) {
        const i64 lhs = static_cast<i64>(a[i]);
        const i64 rhs = static_cast<i64>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}

extern "C" i64 dot_u8_i8_scalar(const u8* a, const i8* b, usize len) {
    i64 sum = 0;
    for (usize i = 0; i < len; ++i) {
        const i64 lhs = static_cast<i64>(a[i]);
        const i64 rhs = static_cast<i64>(b[i]);
        sum += lhs * rhs;
    }
    return sum;
}
