use simd_lab_rust::{
    axpy_scalar, dispatch_tier, dot_f32_scalar, dot_f64_scalar, dot_i16_scalar, dot_u8_i8_scalar,
    f16c::clamp_f16c, sad_u8_best, sad_u8_scalar, sat_add_u8_best, sat_add_u8_dispatch_tier,
    sat_add_u8_scalar, squared_error_best, squared_error_scalar, widen_mul_i16_i32_scalar,
    widen_mul_i8_i16_scalar, widen_mul_u16_u32_scalar, widen_mul_u8_u16_scalar,
};
use std::hint::black_box;
use std::time::Instant;

const SIZES: [usize; 6] = [1 << 10, 1 << 13, 1 << 16, 1 << 18, 1 << 20, 1 << 22];
const WARMUP_SAMPLES: usize = 3;
const SAMPLE_COUNT: usize = 15;
const TARGET_ELEMENTS_PER_SAMPLE: usize = 1 << 20;
const MAX_ITERATIONS_PER_SAMPLE: usize = 4096;
const HALF_VALUES: [u16; 8] = [
    0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400,
];

struct Measurement {
    iterations_per_sample: usize,
    ns_per_element: Vec<f64>,
}

struct Summary {
    min: f64,
    median: f64,
    p95: f64,
    mad: f64,
}

fn iterations_per_sample(n: usize) -> usize {
    (TARGET_ELEMENTS_PER_SAMPLE / n)
        .max(1)
        .min(MAX_ITERATIONS_PER_SAMPLE)
}

fn measure(n: usize, mut run: impl FnMut()) -> Measurement {
    let iterations = iterations_per_sample(n);
    for _ in 0..WARMUP_SAMPLES {
        for _ in 0..iterations {
            run();
        }
    }

    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let start = Instant::now();
        for _ in 0..iterations {
            run();
        }
        let elapsed_ns = start.elapsed().as_secs_f64() * 1e9;
        samples.push(elapsed_ns / (iterations * n) as f64);
    }

    Measurement {
        iterations_per_sample: iterations,
        ns_per_element: samples,
    }
}

fn median(sorted: &[f64]) -> f64 {
    sorted[sorted.len() / 2]
}

fn summarize(values: &[f64]) -> Summary {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median_value = median(&sorted);
    let mut deviations: Vec<f64> = sorted
        .iter()
        .map(|value| (value - median_value).abs())
        .collect();
    deviations.sort_by(f64::total_cmp);
    let p95_index = ((sorted.len() * 95).div_ceil(100)).saturating_sub(1);

    Summary {
        min: sorted[0],
        median: median_value,
        p95: sorted[p95_index],
        mad: median(&deviations),
    }
}

fn report(
    name: &str,
    n: usize,
    working_set_bytes: usize,
    effective_bytes_per_iteration: usize,
    measurement: &Measurement,
) {
    let summary = summarize(&measurement.ns_per_element);
    let bytes_per_element = effective_bytes_per_iteration as f64 / n as f64;
    let gib_s = bytes_per_element / (summary.median * 1e-9) / (1_u64 << 30) as f64;

    print!(
        "RESULT name={name} n={n} working_set_bytes={working_set_bytes} \
         effective_bytes_per_iteration={effective_bytes_per_iteration} \
         iterations_per_sample={} sample_count={} min_ns_per_element={:.9} \
         median_ns_per_element={:.9} p95_ns_per_element={:.9} \
         mad_ns_per_element={:.9} median_gib_per_second={:.6} raw_ns_per_element=",
        measurement.iterations_per_sample,
        measurement.ns_per_element.len(),
        summary.min,
        summary.median,
        summary.p95,
        summary.mad,
        gib_s,
    );
    for (index, value) in measurement.ns_per_element.iter().enumerate() {
        if index != 0 {
            print!(",");
        }
        print!("{value:.9}");
    }
    println!();
}

fn main() {
    for n in SIZES {
        {
            let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();
            let y: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.0005).collect();
            let mut dst = vec![0.0_f32; n];

            let measurement = measure(n, || {
                axpy_scalar(
                    black_box(&mut dst),
                    black_box(&x),
                    black_box(&y),
                    black_box(0.75),
                );
            });
            report("axpy/scalar-autovec", n, n * 12, n * 12, &measurement);

            let measurement = measure(n, || {
                black_box(squared_error_scalar(black_box(&x), black_box(&y)));
            });
            report("sqerr/scalar-f64", n, n * 8, n * 8, &measurement);

            let measurement = measure(n, || {
                black_box(squared_error_best(black_box(&x), black_box(&y)));
            });
            report("sqerr/best-dispatch-f64", n, n * 8, n * 8, &measurement);

            let expected_f32_dot: f64 = x
                .iter()
                .zip(&y)
                .map(|(&lhs, &rhs)| f64::from(lhs) * f64::from(rhs))
                .sum();
            assert_eq!(dot_f32_scalar(&x, &y), expected_f32_dot);
            let f64_x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
            let f64_y: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.0005).collect();
            let expected_f64_dot: f64 = f64_x.iter().zip(&f64_y).map(|(&a, &b)| a * b).sum();
            assert_eq!(dot_f64_scalar(&f64_x, &f64_y), expected_f64_dot);

            let measurement = measure(n, || {
                black_box(dot_f32_scalar(black_box(&x), black_box(&y)));
            });
            report("dot-f32/scalar-f64", n, n * 8, n * 8, &measurement);

            let measurement = measure(n, || {
                black_box(dot_f64_scalar(black_box(&f64_x), black_box(&f64_y)));
            });
            report("dot-f64/scalar-f64", n, n * 16, n * 16, &measurement);
        }

        {
            let a: Vec<u8> = (0..n).map(|i| ((i * 17 + 3) & 255) as u8).collect();
            let b: Vec<u8> = (0..n).map(|i| ((i * 29 + 11) & 255) as u8).collect();
            assert_eq!(sad_u8_scalar(&a, &b), sad_u8_best(&a, &b));
            let mut sat_reference = vec![0_u8; n];
            let mut sat_dst = vec![0_u8; n];
            sat_add_u8_scalar(&mut sat_reference, &a, &b);
            sat_add_u8_best(&mut sat_dst, &a, &b);
            assert_eq!(sat_dst, sat_reference);

            let i16_a: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(17)).collect();
            let i16_b: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(29)).collect();
            let expected_i16_dot: i64 = i16_a
                .iter()
                .zip(&i16_b)
                .map(|(&lhs, &rhs)| i64::from(lhs) * i64::from(rhs))
                .sum();
            assert_eq!(dot_i16_scalar(&i16_a, &i16_b), expected_i16_dot);

            let i8_b: Vec<i8> = (0..n).map(|i| (i as i8).wrapping_mul(3)).collect();
            let expected_mixed_dot: i64 = a
                .iter()
                .zip(&i8_b)
                .map(|(&lhs, &rhs)| i64::from(lhs) * i64::from(rhs))
                .sum();
            assert_eq!(dot_u8_i8_scalar(&a, &i8_b), expected_mixed_dot);

            let u16_a: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(257)).collect();
            let u16_b: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(13)).collect();
            let i16_widen_a = i16_a.clone();
            let i16_widen_b = i16_b.clone();
            let mut u16_dst = vec![0_u16; n];
            let mut i16_dst = vec![0_i16; n];
            let mut u32_dst = vec![0_u32; n];
            let mut i32_dst = vec![0_i32; n];
            let u16_expected: Vec<u16> = a
                .iter()
                .zip(&b)
                .map(|(&lhs, &rhs)| u16::from(lhs) * u16::from(rhs))
                .collect();
            let i16_expected: Vec<i16> = i8_b
                .iter()
                .zip(&i8_b)
                .map(|(&lhs, &rhs)| i16::from(lhs) * i16::from(rhs))
                .collect();
            let u32_expected: Vec<u32> = u16_a
                .iter()
                .zip(&u16_b)
                .map(|(&lhs, &rhs)| u32::from(lhs) * u32::from(rhs))
                .collect();
            let i32_expected: Vec<i32> = i16_widen_a
                .iter()
                .zip(&i16_widen_b)
                .map(|(&lhs, &rhs)| i32::from(lhs) * i32::from(rhs))
                .collect();
            widen_mul_u8_u16_scalar(&mut u16_dst, &a, &b);
            widen_mul_i8_i16_scalar(&mut i16_dst, &i8_b, &i8_b);
            widen_mul_u16_u32_scalar(&mut u32_dst, &u16_a, &u16_b);
            widen_mul_i16_i32_scalar(&mut i32_dst, &i16_widen_a, &i16_widen_b);
            assert_eq!(u16_dst, u16_expected);
            assert_eq!(i16_dst, i16_expected);
            assert_eq!(u32_dst, u32_expected);
            assert_eq!(i32_dst, i32_expected);
            let measurement = measure(n, || {
                black_box(sad_u8_scalar(black_box(&a), black_box(&b)));
            });
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                black_box(sad_u8_best(black_box(&a), black_box(&b)));
            });
            report("sad-u8/best-dispatch", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                sat_add_u8_scalar(black_box(&mut sat_dst), black_box(&a), black_box(&b));
                black_box(&sat_dst);
            });
            report("sat-add-u8/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                sat_add_u8_best(black_box(&mut sat_dst), black_box(&a), black_box(&b));
                black_box(&sat_dst);
            });
            report("sat-add-u8/best-dispatch", n, n * 3, n * 3, &measurement);
            let measurement = measure(n, || {
                black_box(dot_i16_scalar(black_box(&i16_a), black_box(&i16_b)));
            });
            report("dot-i16/scalar-i64", n, n * 4, n * 4, &measurement);

            let measurement = measure(n, || {
                black_box(dot_u8_i8_scalar(black_box(&a), black_box(&i8_b)));
            });
            report("dot-u8-i8/scalar-i64", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                widen_mul_u8_u16_scalar(black_box(&mut u16_dst), black_box(&a), black_box(&b));
                black_box(&u16_dst);
            });
            report(
                "widen-mul-u8-u16/scalar-autovec",
                n,
                n * 4,
                n * 4,
                &measurement,
            );

            let measurement = measure(n, || {
                widen_mul_i8_i16_scalar(
                    black_box(&mut i16_dst),
                    black_box(&i8_b),
                    black_box(&i8_b),
                );
                black_box(&i16_dst);
            });
            report(
                "widen-mul-i8-i16/scalar-autovec",
                n,
                n * 4,
                n * 4,
                &measurement,
            );

            let measurement = measure(n, || {
                widen_mul_u16_u32_scalar(
                    black_box(&mut u32_dst),
                    black_box(&u16_a),
                    black_box(&u16_b),
                );
                black_box(&u32_dst);
            });
            report(
                "widen-mul-u16-u32/scalar-autovec",
                n,
                n * 8,
                n * 8,
                &measurement,
            );

            let measurement = measure(n, || {
                widen_mul_i16_i32_scalar(
                    black_box(&mut i32_dst),
                    black_box(&i16_widen_a),
                    black_box(&i16_widen_b),
                );
                black_box(&i32_dst);
            });
            report(
                "widen-mul-i16-i32/scalar-autovec",
                n,
                n * 8,
                n * 8,
                &measurement,
            );
        }

        {
            let c: Vec<u16> = (0..n).map(|i| HALF_VALUES[i & 7]).collect();
            let lo = vec![0x3800_u16; n];
            let hi = vec![0x4000_u16; n];
            let mut dst = vec![0_u16; n];

            if clamp_f16c(&mut dst, &c, &lo, &hi) {
                let expected: Vec<u16> =
                    c.iter().map(|&value| value.clamp(0x3800, 0x4000)).collect();
                assert_eq!(dst, expected);

                let measurement = measure(n, || {
                    assert!(clamp_f16c(
                        black_box(&mut dst),
                        black_box(&c),
                        black_box(&lo),
                        black_box(&hi),
                    ));
                    black_box(&dst);
                });
                report("clamp-f16/f16c-f32", n, n * 8, n * 8, &measurement);
            } else {
                println!("SKIP name=clamp-f16/f16c-f32 n={n} reason=avx-f16c-unavailable");
            }
        }
    }

    println!(
        "META size_count={} warmup_samples={WARMUP_SAMPLES} sample_count={SAMPLE_COUNT} \
         target_elements_per_sample={TARGET_ELEMENTS_PER_SAMPLE} dispatch_tier={} \
         sat_add_u8_dispatch_tier={}",
        SIZES.len(),
        dispatch_tier(),
        sat_add_u8_dispatch_tier(),
    );
}
