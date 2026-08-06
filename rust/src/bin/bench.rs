use simd_lab_rust::{axpy_scalar, f16c::clamp_f16c, squared_error_best, squared_error_scalar};
use std::hint::black_box;
use std::time::{Duration, Instant};

const N: usize = 1 << 20;
const WARMUP: usize = 8;
const ITERS: usize = 64;
const HALF_VALUES: [u16; 8] = [0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400];

fn measure(mut f: impl FnMut()) -> Duration {
    for _ in 0..WARMUP { f(); }
    let start = Instant::now();
    for _ in 0..ITERS { f(); }
    start.elapsed()
}

fn report(name: &str, elapsed: Duration, bytes_per_iter: usize) {
    let ns_iter = elapsed.as_nanos() as f64 / ITERS as f64;
    let ns_elem = ns_iter / N as f64;
    let gib_s = (bytes_per_iter as f64 * ITERS as f64) / elapsed.as_secs_f64() / (1u64 << 30) as f64;
    println!("{name:28} {ns_elem:9.4} ns/elem  {gib_s:8.2} GiB/s");
}

fn main() {
    let x: Vec<f32> = (0..N).map(|i| i as f32 * 0.001).collect();
    let y: Vec<f32> = (0..N).map(|i| 1.0 + i as f32 * 0.0005).collect();
    let mut dst = vec![0.0_f32; N];

    let t = measure(|| axpy_scalar(black_box(&mut dst), black_box(&x), black_box(&y), black_box(0.75)));
    report("axpy/scalar-autovec", t, N * 12);

    let t = measure(|| { black_box(squared_error_scalar(black_box(&x), black_box(&y))); });
    report("sqerr/scalar-autovec", t, N * 8);

    let t = measure(|| { black_box(squared_error_best(black_box(&x), black_box(&y))); });
    report("sqerr/best-dispatch", t, N * 8);

    let c: Vec<u16> = (0..N).map(|i| HALF_VALUES[i & 7]).collect();
    let lo = vec![0x3800_u16; N]; // 0.5
    let hi = vec![0x4000_u16; N]; // 2.0
    let mut half_dst = vec![0_u16; N];

    if clamp_f16c(&mut half_dst, &c, &lo, &hi) {
        let expected: Vec<u16> = c.iter().map(|&v| v.clamp(0x3800, 0x4000)).collect();
        assert_eq!(half_dst, expected);
        let t = measure(|| {
            assert!(clamp_f16c(
                black_box(&mut half_dst),
                black_box(&c),
                black_box(&lo),
                black_box(&hi),
            ));
            black_box(&half_dst);
        });
        report("clamp-f16/f16c-f32", t, N * 8);
    } else {
        println!("clamp-f16/f16c-f32         skipped (AVX+F16C unavailable)");
    }

    println!("N={N} warmup={WARMUP} iterations={ITERS}");
}
