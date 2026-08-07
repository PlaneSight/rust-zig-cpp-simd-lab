use simd_lab_rust::{axpy_scalar, squared_error_best, squared_error_scalar};

fn main() {
    let n = 1 << 20;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.0005).collect();
    let mut dst = vec![0.0_f32; n];

    axpy_scalar(&mut dst, &x, &y, 0.75);
    let scalar = squared_error_scalar(&x, &y);
    let best = squared_error_best(&x, &y);

    println!("Rust SIMD lab smoke test");
    println!("AXPY checksum: {:.6}", dst.iter().map(|&v| v as f64).sum::<f64>());
    println!("Squared error scalar: {scalar:.6}");
    println!("Squared error best:   {best:.6}");
}
