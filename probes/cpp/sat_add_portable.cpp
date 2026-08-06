// Architecture-neutral u8 saturating-add probe for cross-target autovectorization.
using u8 = unsigned char;
using usize = __SIZE_TYPE__;

extern "C" void sat_add_u8_portable(u8* dst, const u8* a, const u8* b, usize len) {
    for (usize i = 0; i < len; ++i) {
        const unsigned sum = static_cast<unsigned>(a[i]) + static_cast<unsigned>(b[i]);
        dst[i] = static_cast<u8>(sum > 255u ? 255u : sum);
    }
}
