// Architecture-neutral u8 SAD probe for autovectorization experiments.
// Deliberately avoids ISA headers and C++ library dependencies so Clang can
// cross-compile it for AArch64 and wasm32 without a target sysroot.

using u8 = unsigned char;
using u64 = unsigned long long;
using usize = __SIZE_TYPE__;

extern "C" u64 sad_u8_portable(const u8* a, const u8* b, usize len) {
    u64 sum = 0;
    for (usize i = 0; i < len; ++i) {
        const u8 x = a[i];
        const u8 y = b[i];
        sum += x > y ? static_cast<u64>(x - y)
                     : static_cast<u64>(y - x);
    }
    return sum;
}
