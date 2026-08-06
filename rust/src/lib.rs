pub mod f16c;

pub fn axpy_scalar(dst: &mut [f32], x: &[f32], y: &[f32], a: f32) {
    assert_eq!(dst.len(), x.len());
    assert_eq!(x.len(), y.len());

    for ((d, &xv), &yv) in dst.iter_mut().zip(x).zip(y) {
        *d = a.mul_add(xv, yv);
    }
}

pub fn squared_error_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn squared_error_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    assert_eq!(a.len(), b.len());

    let mut i = 0;
    let mut acc = _mm256_setzero_ps();

    while i + 8 <= a.len() {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += 8;
    }

    let mut lanes = [0.0_f32; 8];
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum: f32 = lanes.into_iter().sum();

    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }

    sum
}

pub fn squared_error_best(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { squared_error_avx2(a, b) };
        }
    }

    squared_error_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axpy_matches_expected() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];
        let mut dst = [0.0; 4];
        axpy_scalar(&mut dst, &x, &y, 2.0);
        assert_eq!(dst, [7.0, 10.0, 13.0, 16.0]);
    }

    #[test]
    fn best_squared_error_matches_scalar() {
        let a: Vec<f32> = (0..257).map(|x| x as f32 * 0.25).collect();
        let b: Vec<f32> = (0..257).map(|x| x as f32 * 0.125 + 1.0).collect();
        let scalar = squared_error_scalar(&a, &b);
        let best = squared_error_best(&a, &b);
        let tolerance = scalar.abs().max(1.0) * 1e-5;
        assert!((scalar - best).abs() <= tolerance, "{scalar} != {best}");
    }
}
