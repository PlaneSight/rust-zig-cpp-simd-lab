use simd_lab_rust::f16c::clamp_f16c;

const CASES: [u16; 24] = [
    0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x37ff, 0x3800, 0x3801,
    0x3bff, 0x3c00, 0x3c01, 0x4000, 0x7bff, 0x7c00, 0x7e01, 0x7fff,
    0x8001, 0xb800, 0xbc00, 0xc000, 0xfbff, 0xfc00, 0xfe01, 0x0000,
];
const COUNT: usize = 23;

fn main() {
    let lo = [0xb800_u16; 24]; // -0.5
    let hi = [0x4000_u16; 24]; // +2.0
    let mut out = [0_u16; 24];
    let available = clamp_f16c(&mut out, &CASES, &lo, &hi);

    print!("{{\"strategy\":\"rust-f16c\",\"available\":{available},\"outputs\":[");
    if available {
        for (i, bits) in out[..COUNT].iter().enumerate() {
            if i != 0 { print!(","); }
            print!("\"{bits:04x}\"");
        }
    }
    println!("]}}");
}
