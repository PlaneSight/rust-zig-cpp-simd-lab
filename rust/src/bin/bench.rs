use simd_lab_rust::{
    axpy_scalar, dispatch_tier, f16c::clamp_f16c, sad_u8_best, sad_u8_scalar,
    sat_add_u8_best, sat_add_u8_dispatch_tier, sat_add_u8_scalar,
    squared_error_best, squared_error_scalar,
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

            let measurement = measure(n, || {
                black_box(sad_u8_scalar(black_box(&a), black_box(&b)));
            });
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                black_box(sad_u8_best(black_box(&a), black_box(&b)));
            });
            report("sad-u8/best-dispatch", n, n * 2, n * 2, &measurement);

            let measurement = measure(n, || {
                sat_add_u8_scalar(
                    black_box(&mut sat_dst),
                    black_box(&a),
                    black_box(&b),
                );
                black_box(&sat_dst);
            });
            report(
                "sat-add-u8/scalar-autovec",
                n,
                n * 3,
                n * 3,
                &measurement,
            );

            let measurement = measure(n, || {
                sat_add_u8_best(
                    black_box(&mut sat_dst),
                    black_box(&a),
                    black_box(&b),
                );
                black_box(&sat_dst);
            });
            report(
                "sat-add-u8/best-dispatch",
                n,
                n * 3,
                n * 3,
                &measurement,
            );
        }

        {
            let c: Vec<u16> = (0..n).map(|i| HALF_VALUES[i & 7]).collect();
            let lo = vec![0x3800_u16; n];
            let hi = vec![0x4000_u16; n];
            let mut dst = vec![0_u16; n];

            if clamp_f16c(&mut dst, &c, &lo, &hi) {
                let expected: Vec<u16> = c
                    .iter()
                    .map(|&value| value.clamp(0x3800, 0x4000))
                    .collect();
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
                println!(
                    "SKIP name=clamp-f16/f16c-f32 n={n} reason=avx-f16c-unavailable"
                );
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
