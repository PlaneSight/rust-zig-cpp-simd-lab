// Clang extension probe for native half arithmetic lowering.
// `_Float16` is intentionally labelled as a compiler extension, not C++23.

using usize = __SIZE_TYPE__;

extern "C" void clamp_f16_ext(_Float16* dst,
                              const _Float16* c,
                              const _Float16* lo,
                              const _Float16* hi,
                              usize len) {
    for (usize i = 0; i < len; ++i) {
        const _Float16 x = c[i];
        const _Float16 l = lo[i];
        const _Float16 h = hi[i];
        dst[i] = x < l ? l : (x > h ? h : x);
    }
}
