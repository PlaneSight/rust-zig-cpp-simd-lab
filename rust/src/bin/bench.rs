use simd_lab_rust::{
    axpy_scalar, blend_u8_scalar, convolve3_u8_scalar, convolve5_u8_scalar,
    convert_f32_u8_round_scalar, convert_f32_u8_sat_scalar,
    convert_f32_u8_trunc_scalar, convert_i16_to_f32_scalar, convert_u16_to_f32_scalar,
    convert_u8_f32_affine_scalar, dispatch_tier, dot_f32_scalar, dot_f64_scalar, dot_i16_scalar,
    dot_u8_i8_scalar, f16c::clamp_f16c, f32_to_u16_sat_scalar, narrow_u16_to_u8_round_scalar,
    narrow_u16_to_u8_sat_scalar, narrow_u16_to_u8_trunc_scalar, pack_u8x4_to_u32_scalar,
    sad_u8_best, sad_u8_scalar, sat_add_u8_best, sat_add_u8_dispatch_tier, sat_add_u8_scalar,
    squared_error_best, squared_error_scalar, unpack_u32_to_u8x4_scalar,
    widen_i16_to_i32_scalar, widen_i8_to_i16_scalar, widen_mul_i16_i32_scalar,
    widen_mul_i32_i64_scalar, widen_mul_i8_i16_scalar, widen_mul_u16_u32_scalar,
    widen_mul_u32_u64_scalar, widen_mul_u8_u16_scalar, widen_u16_to_u32_scalar,
    widen_u8_to_u16_scalar, widen_u8_to_u32_scalar,
};
use simd_lab_rust::{
    sat_add_i16_scalar, sat_add_i32_scalar, sat_add_i64_scalar, sat_add_i8_scalar,
    sat_add_u16_scalar, sat_add_u32_scalar, sat_add_u64_scalar,
};
const SAT_ADD_VALIDATION_LENGTHS: [usize; 14] =
    [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127];
const SAT_ADD_I8_A: [i8; 8] = [i8::MIN, i8::MAX, i8::MIN + 1, i8::MAX - 1, -1, 0, 37, -73];
const SAT_ADD_I8_B: [i8; 8] = [i8::MIN, i8::MAX, i8::MAX, i8::MIN, 1, -1, -44, 68];
const SAT_ADD_U16_A: [u16; 8] = [
    0,
    u16::MAX,
    1,
    u16::MAX - 1,
    0x1234,
    0x8000,
    0xdead,
    0x55aa,
];
const SAT_ADD_U16_B: [u16; 8] = [
    u16::MAX,
    u16::MAX,
    1,
    u16::MAX,
    0xedcb,
    0x7fff,
    0x1234,
    0xaa55,
];
const SAT_ADD_I16_A: [i16; 8] = [
    i16::MIN,
    i16::MAX,
    i16::MIN + 1,
    i16::MAX - 1,
    -1,
    0,
    12_345,
    -23_456,
];
const SAT_ADD_I16_B: [i16; 8] = [i16::MIN, i16::MAX, i16::MAX, i16::MIN, 1, -1, -12_345, 23_456];
const SAT_ADD_U32_A: [u32; 8] = [
    0,
    u32::MAX,
    1,
    u32::MAX - 1,
    0x1234_5678,
    0x8000_0000,
    0xdead_beef,
    0x55aa_55aa,
];
const SAT_ADD_U32_B: [u32; 8] = [
    u32::MAX,
    u32::MAX,
    1,
    u32::MAX,
    0xedcb_a987,
    0x7fff_ffff,
    0x1234_5678,
    0xaa55_aa55,
];
const SAT_ADD_I32_A: [i32; 8] = [
    i32::MIN,
    i32::MAX,
    i32::MIN + 1,
    i32::MAX - 1,
    -1,
    0,
    0x1234_5678,
    -0x2345_6789,
];
const SAT_ADD_I32_B: [i32; 8] = [
    i32::MIN,
    i32::MAX,
    i32::MAX,
    i32::MIN,
    1,
    -1,
    -0x1234_5678,
    0x2345_6789,
];
const SAT_ADD_U64_A: [u64; 8] = [
    0,
    u64::MAX,
    1,
    u64::MAX - 1,
    0x1234_5678_9abc_def0,
    0x8000_0000_0000_0000,
    0xdead_beef_cafe_babe,
    0x55aa_55aa_aa55_55aa,
];
const SAT_ADD_U64_B: [u64; 8] = [
    u64::MAX,
    u64::MAX,
    1,
    u64::MAX,
    0xedcb_a987_6543_210f,
    0x7fff_ffff_ffff_ffff,
    0x1234_5678_9abc_def0,
    0xaa55_aa55_55aa_aa55,
];
const SAT_ADD_I64_A: [i64; 8] = [
    i64::MIN,
    i64::MAX,
    i64::MIN + 1,
    i64::MAX - 1,
    -1,
    0,
    0x1234_5678_9abc_def0,
    -0x2345_6789_abcd_ef01,
];
const SAT_ADD_I64_B: [i64; 8] = [
    i64::MIN,
    i64::MAX,
    i64::MAX,
    i64::MIN,
    1,
    -1,
    -0x1234_5678_9abc_def0,
    0x2345_6789_abcd_ef01,
];
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
            let mut blend_dst = vec![0_u8; n];
            let mut convolve3_dst = vec![0_u8; n];
            let mut convolve5_dst = vec![0_u8; n];
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

            let sat_i8_a: Vec<i8> = (0..n).map(|i| SAT_ADD_I8_A[i & 7]).collect();
            let sat_i8_b: Vec<i8> = (0..n).map(|i| SAT_ADD_I8_B[i & 7]).collect();
            let sat_u16_a: Vec<u16> = (0..n).map(|i| SAT_ADD_U16_A[i & 7]).collect();
            let sat_u16_b: Vec<u16> = (0..n).map(|i| SAT_ADD_U16_B[i & 7]).collect();
            let sat_i16_a: Vec<i16> = (0..n).map(|i| SAT_ADD_I16_A[i & 7]).collect();
            let sat_i16_b: Vec<i16> = (0..n).map(|i| SAT_ADD_I16_B[i & 7]).collect();
            let sat_u32_a: Vec<u32> = (0..n).map(|i| SAT_ADD_U32_A[i & 7]).collect();
            let sat_u32_b: Vec<u32> = (0..n).map(|i| SAT_ADD_U32_B[i & 7]).collect();
            let sat_i32_a: Vec<i32> = (0..n).map(|i| SAT_ADD_I32_A[i & 7]).collect();
            let sat_i32_b: Vec<i32> = (0..n).map(|i| SAT_ADD_I32_B[i & 7]).collect();
            let sat_u64_a: Vec<u64> = (0..n).map(|i| SAT_ADD_U64_A[i & 7]).collect();
            let sat_u64_b: Vec<u64> = (0..n).map(|i| SAT_ADD_U64_B[i & 7]).collect();
            let sat_i64_a: Vec<i64> = (0..n).map(|i| SAT_ADD_I64_A[i & 7]).collect();
            let sat_i64_b: Vec<i64> = (0..n).map(|i| SAT_ADD_I64_B[i & 7]).collect();
            let mut sat_i8_dst = vec![0_i8; n];
            let mut sat_u16_dst = vec![0_u16; n];
            let mut sat_i16_dst = vec![0_i16; n];
            let mut sat_u32_dst = vec![0_u32; n];
            let mut sat_i32_dst = vec![0_i32; n];
            let mut sat_u64_dst = vec![0_u64; n];
            let mut sat_i64_dst = vec![0_i64; n];

            let u16_a: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(257)).collect();
            let u16_b: Vec<u16> = (0..n).map(|i| (i as u16).wrapping_mul(13)).collect();
            let u32_a: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(65_537)).collect();
            let u32_b: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(257)).collect();
            let i32_widen_a: Vec<i32> = (0..n).map(|i| (i as i32).wrapping_mul(65_537)).collect();
            let i32_widen_b: Vec<i32> = (0..n).map(|i| (i as i32).wrapping_mul(257)).collect();
            let i16_widen_a = i16_a.clone();
            let i16_widen_b = i16_b.clone();
            let mut u16_dst = vec![0_u16; n];
            let mut i16_dst = vec![0_i16; n];
            let mut u32_dst = vec![0_u32; n];
            let mut i32_dst = vec![0_i32; n];
            let mut u64_dst = vec![0_u64; n];
            let mut i64_dst = vec![0_i64; n];
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
            let u64_expected: Vec<u64> = u32_a
                .iter()
                .zip(&u32_b)
                .map(|(&lhs, &rhs)| u64::from(lhs) * u64::from(rhs))
                .collect();
            let i64_expected: Vec<i64> = i32_widen_a
                .iter()
                .zip(&i32_widen_b)
                .map(|(&lhs, &rhs)| i64::from(lhs) * i64::from(rhs))
                .collect();
            widen_mul_u8_u16_scalar(&mut u16_dst, &a, &b);
            widen_mul_i8_i16_scalar(&mut i16_dst, &i8_b, &i8_b);
            widen_mul_u16_u32_scalar(&mut u32_dst, &u16_a, &u16_b);
            widen_mul_i16_i32_scalar(&mut i32_dst, &i16_widen_a, &i16_widen_b);
            widen_mul_u32_u64_scalar(&mut u64_dst, &u32_a, &u32_b);
            widen_mul_i32_i64_scalar(&mut i64_dst, &i32_widen_a, &i32_widen_b);
            assert_eq!(u64_dst, u64_expected);
            assert_eq!(i64_dst, i64_expected);
            assert_eq!(u16_dst, u16_expected);
            assert_eq!(i16_dst, i16_expected);
            assert_eq!(u32_dst, u32_expected);
            assert_eq!(i32_dst, i32_expected);
            let blend_weight = 77_u16;
            let blend_weight_u32 = u32::from(blend_weight);
            let blend_expected: Vec<u8> = a
                .iter()
                .zip(&b)
                .map(|(&lhs, &rhs)| {
                    ((u32::from(lhs) * (256 - blend_weight_u32)
                        + u32::from(rhs) * blend_weight_u32
                        + 128)
                        >> 8) as u8
                })
                .collect();
            blend_u8_scalar(&mut blend_dst, &a, &b, blend_weight);
            assert_eq!(blend_dst, blend_expected);

            let mut convolve3_expected = vec![0_u8; n];
            let mut convolve5_expected = vec![0_u8; n];
            if n != 0 {
                let last = n - 1;
                for i in 0..n {
                    let left = a[i.saturating_sub(1)];
                    let right = a[i.saturating_add(1).min(last)];
                    convolve3_expected[i] =
                        ((u32::from(left) + 2 * u32::from(a[i]) + u32::from(right) + 2) >> 2)
                            as u8;

                    let left2 = a[i.saturating_sub(2)];
                    let right2 = a[i.saturating_add(2).min(last)];
                    convolve5_expected[i] = ((
                        u32::from(left2)
                            + 4 * u32::from(left)
                            + 6 * u32::from(a[i])
                            + 4 * u32::from(right)
                            + u32::from(right2)
                            + 8
                    ) >> 4) as u8;
                }
            }
            convolve3_u8_scalar(&mut convolve3_dst, &a);
            convolve5_u8_scalar(&mut convolve5_dst, &a);
            assert_eq!(convolve3_dst, convolve3_expected);
            assert_eq!(convolve5_dst, convolve5_expected);

            let measurement = measure(n, || {
                black_box(sad_u8_scalar(black_box(&a), black_box(&b)));
            });
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                black_box(sad_u8_best(black_box(&a), black_box(&b)));
            });
            report("sad-u8/best-dispatch", n, n * 2, n * 2, &measurement);
            let measurement = measure(n, || {
                blend_u8_scalar(
                    black_box(&mut blend_dst),
                    black_box(&a),
                    black_box(&b),
                    black_box(blend_weight),
                );
                black_box(&blend_dst);
            });
            report("blend-u8/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                convolve3_u8_scalar(black_box(&mut convolve3_dst), black_box(&a));
                black_box(&convolve3_dst);
            });
            report("convolve3-u8/scalar-autovec", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                convolve5_u8_scalar(black_box(&mut convolve5_dst), black_box(&a));
                black_box(&convolve5_dst);
            });
            report("convolve5-u8/scalar-autovec", n, n * 2, n * 2, &measurement);


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
            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<i8> = (0..len).map(|i| SAT_ADD_I8_A[i & 7]).collect();
                    let b: Vec<i8> = (0..len).map(|i| SAT_ADD_I8_B[i & 7]).collect();
                    let expected: Vec<i8> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (i16::from(x) + i16::from(y))
                                .clamp(i16::from(i8::MIN), i16::from(i8::MAX))
                                as i8
                        })
                        .collect();
                    let mut candidate = vec![0_i8; len];
                    sat_add_i8_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-i8 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_i8_scalar(
                        black_box(&mut sat_i8_dst),
                        black_box(&sat_i8_a),
                        black_box(&sat_i8_b),
                    );
                    black_box(&sat_i8_dst);
                });
                report("sat-add-i8/scalar-autovec", n, n * 3, n * 3, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<u16> = (0..len).map(|i| SAT_ADD_U16_A[i & 7]).collect();
                    let b: Vec<u16> = (0..len).map(|i| SAT_ADD_U16_B[i & 7]).collect();
                    let expected: Vec<u16> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (u32::from(x) + u32::from(y)).min(u32::from(u16::MAX)) as u16
                        })
                        .collect();
                    let mut candidate = vec![0_u16; len];
                    sat_add_u16_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-u16 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_u16_scalar(
                        black_box(&mut sat_u16_dst),
                        black_box(&sat_u16_a),
                        black_box(&sat_u16_b),
                    );
                    black_box(&sat_u16_dst);
                });
                report("sat-add-u16/scalar-autovec", n, n * 6, n * 6, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<i16> = (0..len).map(|i| SAT_ADD_I16_A[i & 7]).collect();
                    let b: Vec<i16> = (0..len).map(|i| SAT_ADD_I16_B[i & 7]).collect();
                    let expected: Vec<i16> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (i32::from(x) + i32::from(y))
                                .clamp(i32::from(i16::MIN), i32::from(i16::MAX))
                                as i16
                        })
                        .collect();
                    let mut candidate = vec![0_i16; len];
                    sat_add_i16_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-i16 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_i16_scalar(
                        black_box(&mut sat_i16_dst),
                        black_box(&sat_i16_a),
                        black_box(&sat_i16_b),
                    );
                    black_box(&sat_i16_dst);
                });
                report("sat-add-i16/scalar-autovec", n, n * 6, n * 6, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<u32> = (0..len).map(|i| SAT_ADD_U32_A[i & 7]).collect();
                    let b: Vec<u32> = (0..len).map(|i| SAT_ADD_U32_B[i & 7]).collect();
                    let expected: Vec<u32> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (u64::from(x) + u64::from(y)).min(u64::from(u32::MAX)) as u32
                        })
                        .collect();
                    let mut candidate = vec![0_u32; len];
                    sat_add_u32_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-u32 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_u32_scalar(
                        black_box(&mut sat_u32_dst),
                        black_box(&sat_u32_a),
                        black_box(&sat_u32_b),
                    );
                    black_box(&sat_u32_dst);
                });
                report("sat-add-u32/scalar-autovec", n, n * 12, n * 12, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<i32> = (0..len).map(|i| SAT_ADD_I32_A[i & 7]).collect();
                    let b: Vec<i32> = (0..len).map(|i| SAT_ADD_I32_B[i & 7]).collect();
                    let expected: Vec<i32> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (i64::from(x) + i64::from(y))
                                .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                                as i32
                        })
                        .collect();
                    let mut candidate = vec![0_i32; len];
                    sat_add_i32_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-i32 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_i32_scalar(
                        black_box(&mut sat_i32_dst),
                        black_box(&sat_i32_a),
                        black_box(&sat_i32_b),
                    );
                    black_box(&sat_i32_dst);
                });
                report("sat-add-i32/scalar-autovec", n, n * 12, n * 12, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<u64> = (0..len).map(|i| SAT_ADD_U64_A[i & 7]).collect();
                    let b: Vec<u64> = (0..len).map(|i| SAT_ADD_U64_B[i & 7]).collect();
                    let expected: Vec<u64> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (u128::from(x) + u128::from(y)).min(u128::from(u64::MAX)) as u64
                        })
                        .collect();
                    let mut candidate = vec![0_u64; len];
                    sat_add_u64_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-u64 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_u64_scalar(
                        black_box(&mut sat_u64_dst),
                        black_box(&sat_u64_a),
                        black_box(&sat_u64_b),
                    );
                    black_box(&sat_u64_dst);
                });
                report("sat-add-u64/scalar-autovec", n, n * 24, n * 24, &measurement);
            }

            {
                for &len in &SAT_ADD_VALIDATION_LENGTHS {
                    let a: Vec<i64> = (0..len).map(|i| SAT_ADD_I64_A[i & 7]).collect();
                    let b: Vec<i64> = (0..len).map(|i| SAT_ADD_I64_B[i & 7]).collect();
                    // Widen before addition so i64 extrema test saturation, not wrapping.
                    let expected: Vec<i64> = a
                        .iter()
                        .zip(&b)
                        .map(|(&x, &y)| {
                            (i128::from(x) + i128::from(y))
                                .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                                as i64
                        })
                        .collect();
                    let mut candidate = vec![0_i64; len];
                    sat_add_i64_scalar(&mut candidate, &a, &b);
                    assert_eq!(candidate, expected, "sat-add-i64 len={len}");
                }
                let measurement = measure(n, || {
                    sat_add_i64_scalar(
                        black_box(&mut sat_i64_dst),
                        black_box(&sat_i64_a),
                        black_box(&sat_i64_b),
                    );
                    black_box(&sat_i64_dst);
                });
                report("sat-add-i64/scalar-autovec", n, n * 24, n * 24, &measurement);
            }

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
            let measurement = measure(n, || {
                widen_mul_u32_u64_scalar(
                    black_box(&mut u64_dst),
                    black_box(&u32_a),
                    black_box(&u32_b),
                );
                black_box(&u64_dst);
            });
            report(
                "widen-mul-u32-u64/scalar-autovec",
                n,
                n * 16,
                n * 16,
                &measurement,
            );

            let measurement = measure(n, || {
                widen_mul_i32_i64_scalar(
                    black_box(&mut i64_dst),
                    black_box(&i32_widen_a),
                    black_box(&i32_widen_b),
                );
                black_box(&i64_dst);
            });
            report(
                "widen-mul-i32-i64/scalar-autovec",
                n,
                n * 16,
                n * 16,
                &measurement,
            );
        }
        {
            let mixed_u8: Vec<u8> = (0..n)
                .map(|i| ((i.wrapping_mul(37).wrapping_add(11)) & 255) as u8)
                .collect();
            let mixed_i8: Vec<i8> = (0..n)
                .map(|i| (i as i8).wrapping_mul(13).wrapping_add(7))
                .collect();
            let mixed_u16: Vec<u16> = (0..n)
                .map(|i| (i as u16).wrapping_mul(257).wrapping_add(3))
                .collect();
            let mixed_i16: Vec<i16> = (0..n)
                .map(|i| (i as i16).wrapping_mul(521).wrapping_sub(17))
                .collect();
            let mixed_f32_valid: Vec<f32> = (0..n)
                .map(|i| ((i.wrapping_mul(911) % 255_001) as f32) / 1_000.0)
                .collect();
            let mixed_f32_boundary: Vec<f32> = (0..n)
                .map(|i| {
                    [
                        f32::NAN,
                        f32::NEG_INFINITY,
                        -0.5,
                        0.5,
                        1.75,
                        65_534.75,
                        65_535.0,
                        f32::INFINITY,
                    ][i & 7]
                })
                .collect();

            let mut mixed_u16_from_u8 = vec![0_u16; n];
            let mut mixed_u32_from_u8 = vec![0_u32; n];
            let mut mixed_i16_from_i8 = vec![0_i16; n];
            let mut mixed_u32_from_u16 = vec![0_u32; n];
            let mut mixed_i32_from_i16 = vec![0_i32; n];
            widen_u8_to_u16_scalar(&mut mixed_u16_from_u8, &mixed_u8);
            widen_u8_to_u32_scalar(&mut mixed_u32_from_u8, &mixed_u8);
            widen_i8_to_i16_scalar(&mut mixed_i16_from_i8, &mixed_i8);
            widen_u16_to_u32_scalar(&mut mixed_u32_from_u16, &mixed_u16);
            widen_i16_to_i32_scalar(&mut mixed_i32_from_i16, &mixed_i16);
            assert_eq!(
                mixed_u16_from_u8,
                mixed_u8.iter().map(|&x| u16::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_u32_from_u8,
                mixed_u8.iter().map(|&x| u32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_i16_from_i8,
                mixed_i8.iter().map(|&x| i16::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_u32_from_u16,
                mixed_u16.iter().map(|&x| u32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_i32_from_i16,
                mixed_i16.iter().map(|&x| i32::from(x)).collect::<Vec<_>>()
            );

            let mut mixed_f32_from_u16 = vec![0.0_f32; n];
            let mut mixed_f32_from_i16 = vec![0.0_f32; n];
            let mut mixed_affine = vec![0.0_f32; n];
            convert_u16_to_f32_scalar(&mut mixed_f32_from_u16, &mixed_u16);
            convert_i16_to_f32_scalar(&mut mixed_f32_from_i16, &mixed_i16);
            convert_u8_f32_affine_scalar(&mut mixed_affine, &mixed_u8, 1.25, -3.5);
            assert_eq!(
                mixed_f32_from_u16,
                mixed_u16.iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_f32_from_i16,
                mixed_i16.iter().map(|&x| f32::from(x)).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_affine,
                mixed_u8
                    .iter()
                    .map(|&x| f32::from(x) * 1.25 - 3.5)
                    .collect::<Vec<_>>()
            );

            let mut mixed_u16_from_f32 = vec![0_u16; n];
            f32_to_u16_sat_scalar(&mut mixed_u16_from_f32, &mixed_f32_boundary);
            let expected_u16_from_f32: Vec<u16> = mixed_f32_boundary
                .iter()
                .map(|&value| {
                    if value.is_nan() || value <= 0.0 {
                        0
                    } else if value >= f32::from(u16::MAX) {
                        u16::MAX
                    } else {
                        value as u16
                    }
                })
                .collect();
            assert_eq!(mixed_u16_from_f32, expected_u16_from_f32);

            let mut mixed_u8_trunc = vec![0_u8; n];
            let mut mixed_u8_round = vec![0_u8; n];
            let mut mixed_u8_sat = vec![0_u8; n];
            convert_f32_u8_trunc_scalar(&mut mixed_u8_trunc, &mixed_f32_valid);
            convert_f32_u8_round_scalar(&mut mixed_u8_round, &mixed_f32_valid);
            convert_f32_u8_sat_scalar(&mut mixed_u8_sat, &mixed_f32_boundary);
            assert_eq!(
                mixed_u8_trunc,
                mixed_f32_valid.iter().map(|&x| x as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_u8_round,
                mixed_f32_valid
                    .iter()
                    .map(|&x| (x + 0.5).floor() as u8)
                    .collect::<Vec<_>>()
            );
            let expected_u8_sat: Vec<u8> = mixed_f32_boundary
                .iter()
                .map(|&value| {
                    if value.is_nan() || value <= 0.0 {
                        0
                    } else if value >= 255.0 {
                        u8::MAX
                    } else {
                        value as u8
                    }
                })
                .collect();
            assert_eq!(mixed_u8_sat, expected_u8_sat);

            let mut mixed_narrow_trunc = vec![0_u8; n];
            let mut mixed_narrow_round = vec![0_u8; n];
            let mut mixed_narrow_sat = vec![0_u8; n];
            narrow_u16_to_u8_trunc_scalar(&mut mixed_narrow_trunc, &mixed_u16);
            narrow_u16_to_u8_round_scalar(&mut mixed_narrow_round, &mixed_u16);
            narrow_u16_to_u8_sat_scalar(&mut mixed_narrow_sat, &mixed_u16);
            assert_eq!(
                mixed_narrow_trunc,
                mixed_u16.iter().map(|&x| (x & 0xff) as u8).collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_narrow_round,
                mixed_u16
                    .iter()
                    .map(|&x| ((u32::from(x) + 128) / 257) as u8)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                mixed_narrow_sat,
                mixed_u16
                    .iter()
                    .map(|&x| x.min(u16::from(u8::MAX)) as u8)
                    .collect::<Vec<_>>()
            );

            let mixed_pack_src: Vec<u8> = (0..n * 4)
                .map(|i| ((i.wrapping_mul(73).wrapping_add(19)) & 255) as u8)
                .collect();
            let mut mixed_packed = vec![0_u32; n];
            pack_u8x4_to_u32_scalar(&mut mixed_packed, &mixed_pack_src);
            let expected_packed: Vec<u32> = mixed_pack_src
                .chunks_exact(4)
                .map(|chunk| {
                    u32::from(chunk[0])
                        | (u32::from(chunk[1]) << 8)
                        | (u32::from(chunk[2]) << 16)
                        | (u32::from(chunk[3]) << 24)
                })
                .collect();
            assert_eq!(mixed_packed, expected_packed);
            let mut mixed_unpacked = vec![0_u8; n * 4];
            unpack_u32_to_u8x4_scalar(&mut mixed_unpacked, &mixed_packed);
            assert_eq!(mixed_unpacked, mixed_pack_src);

            let measurement = measure(n, || {
                widen_u8_to_u16_scalar(
                    black_box(&mut mixed_u16_from_u8),
                    black_box(&mixed_u8),
                );
                black_box(&mixed_u16_from_u8);
            });
            report("widen-u8-u16/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                widen_u8_to_u32_scalar(
                    black_box(&mut mixed_u32_from_u8),
                    black_box(&mixed_u8),
                );
                black_box(&mixed_u32_from_u8);
            });
            report("widen-u8-u32/scalar-autovec", n, n * 5, n * 5, &measurement);

            let measurement = measure(n, || {
                widen_i8_to_i16_scalar(
                    black_box(&mut mixed_i16_from_i8),
                    black_box(&mixed_i8),
                );
                black_box(&mixed_i16_from_i8);
            });
            report("widen-i8-i16/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                widen_u16_to_u32_scalar(
                    black_box(&mut mixed_u32_from_u16),
                    black_box(&mixed_u16),
                );
                black_box(&mixed_u32_from_u16);
            });
            report("widen-u16-u32/scalar-autovec", n, n * 6, n * 6, &measurement);

            let measurement = measure(n, || {
                widen_i16_to_i32_scalar(
                    black_box(&mut mixed_i32_from_i16),
                    black_box(&mixed_i16),
                );
                black_box(&mixed_i32_from_i16);
            });
            report("widen-i16-i32/scalar-autovec", n, n * 6, n * 6, &measurement);

            let measurement = measure(n, || {
                convert_u16_to_f32_scalar(
                    black_box(&mut mixed_f32_from_u16),
                    black_box(&mixed_u16),
                );
                black_box(&mixed_f32_from_u16);
            });
            report("convert-u16-f32/scalar-autovec", n, n * 6, n * 6, &measurement);

            let measurement = measure(n, || {
                convert_i16_to_f32_scalar(
                    black_box(&mut mixed_f32_from_i16),
                    black_box(&mixed_i16),
                );
                black_box(&mixed_f32_from_i16);
            });
            report("convert-i16-f32/scalar-autovec", n, n * 6, n * 6, &measurement);

            let measurement = measure(n, || {
                convert_u8_f32_affine_scalar(
                    black_box(&mut mixed_affine),
                    black_box(&mixed_u8),
                    black_box(1.25),
                    black_box(-3.5),
                );
                black_box(&mixed_affine);
            });
            report(
                "convert-u8-f32-affine/scalar-autovec",
                n,
                n * 5,
                n * 5,
                &measurement,
            );

            let measurement = measure(n, || {
                f32_to_u16_sat_scalar(
                    black_box(&mut mixed_u16_from_f32),
                    black_box(&mixed_f32_boundary),
                );
                black_box(&mixed_u16_from_f32);
            });
            report("convert-f32-u16-sat/scalar-autovec", n, n * 6, n * 6, &measurement);

            let measurement = measure(n, || {
                convert_f32_u8_trunc_scalar(
                    black_box(&mut mixed_u8_trunc),
                    black_box(&mixed_f32_valid),
                );
                black_box(&mixed_u8_trunc);
            });
            report("convert-f32-u8-trunc/scalar-autovec", n, n * 5, n * 5, &measurement);

            let measurement = measure(n, || {
                convert_f32_u8_round_scalar(
                    black_box(&mut mixed_u8_round),
                    black_box(&mixed_f32_valid),
                );
                black_box(&mixed_u8_round);
            });
            report("convert-f32-u8-round/scalar-autovec", n, n * 5, n * 5, &measurement);

            let measurement = measure(n, || {
                convert_f32_u8_sat_scalar(
                    black_box(&mut mixed_u8_sat),
                    black_box(&mixed_f32_boundary),
                );
                black_box(&mixed_u8_sat);
            });
            report("convert-f32-u8-sat/scalar-autovec", n, n * 5, n * 5, &measurement);

            let measurement = measure(n, || {
                narrow_u16_to_u8_trunc_scalar(
                    black_box(&mut mixed_narrow_trunc),
                    black_box(&mixed_u16),
                );
                black_box(&mixed_narrow_trunc);
            });
            report("narrow-u16-u8-trunc/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                narrow_u16_to_u8_round_scalar(
                    black_box(&mut mixed_narrow_round),
                    black_box(&mixed_u16),
                );
                black_box(&mixed_narrow_round);
            });
            report("narrow-u16-u8-round/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                narrow_u16_to_u8_sat_scalar(
                    black_box(&mut mixed_narrow_sat),
                    black_box(&mixed_u16),
                );
                black_box(&mixed_narrow_sat);
            });
            report("narrow-u16-u8-sat/scalar-autovec", n, n * 3, n * 3, &measurement);

            let measurement = measure(n, || {
                pack_u8x4_to_u32_scalar(
                    black_box(&mut mixed_packed),
                    black_box(&mixed_pack_src),
                );
                black_box(&mixed_packed);
            });
            report("pack-u8x4-u32/scalar-autovec", n, n * 8, n * 8, &measurement);

            let measurement = measure(n, || {
                unpack_u32_to_u8x4_scalar(
                    black_box(&mut mixed_unpacked),
                    black_box(&mixed_packed),
                );
                black_box(&mixed_unpacked);
            });
            report("unpack-u32-u8x4/scalar-autovec", n, n * 8, n * 8, &measurement);
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
