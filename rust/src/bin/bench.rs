use simd_lab_rust::{axpy_scalar, squared_error_best, squared_error_scalar};
use std::hint::black_box;
use std::time::{Duration, Instant};

const N: usize = 1 << 20;
const WARMUP: usize = 8;
const ITERS: usize = 64;

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

    println!("N={N} warmup={WARMUP} iterations={ITERS}");
}
