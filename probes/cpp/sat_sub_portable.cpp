// Architecture-neutral u8 saturating-subtraction probe for cross-target autovectorization.
using u8 = unsigned char;
using usize = __SIZE_TYPE__;

extern "C" void sat_sub_u8_portable(u8* dst, const u8* a, const u8* b,
                                     usize len) {
    for (usize i = 0; i < len; ++i) {
        const unsigned lhs = static_cast<unsigned>(a[i]);
        const unsigned rhs = static_cast<unsigned>(b[i]);
        dst[i] = static_cast<u8>(lhs < rhs ? 0u : lhs - rhs);
    }
}
