// u8/u16 absolute-difference and widening-reduction probes.

pub export fn absdiff_u8x32(a: @Vector(32, u8), b: @Vector(32, u8)) @Vector(32, u8) {
    return @max(a, b) - @min(a, b);
}

pub export fn sad_u8x32(a: @Vector(32, u8), b: @Vector(32, u8)) u16 {
    const diff: @Vector(32, u8) = @max(a, b) - @min(a, b);
    const wide: @Vector(32, u16) = @intCast(diff);
    return @reduce(.Add, wide);
}

pub export fn widen_u8_to_u16x32(a: @Vector(32, u8)) @Vector(32, u16) {
    return @intCast(a);
}

pub export fn sad_u8_scalar(a: [*]const u8, b: [*]const u8, len: usize) u64 {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += @intCast(if (x > y) x - y else y - x);
    }
    return sum;
}

pub export fn sad_u8_vector(a: [*]const u8, b: [*]const u8, len: usize) u64 {
    const Input = @Vector(32, u8);
    const Wide = @Vector(32, u16);
    var sum: u64 = 0;
    var i: usize = 0;

    while (i + 32 <= len) : (i += 32) {
        const va: Input = a[i..][0..32].*;
        const vb: Input = b[i..][0..32].*;
        const difference: Input = @max(va, vb) - @min(va, vb);
        const wide_difference: Wide = @intCast(difference);
        sum += @intCast(@reduce(.Add, wide_difference));
    }

    while (i < len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += @intCast(if (x > y) x - y else y - x);
    }
    return sum;
}

pub export fn sad_u16_scalar(a: [*]const u16, b: [*]const u16, len: usize) u64 {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += @intCast(if (x > y) x - y else y - x);
    }
    return sum;
}

pub export fn sad_u16_vector(a: [*]const u16, b: [*]const u16, len: usize) u64 {
    const Input = @Vector(16, u16);
    const Wide = @Vector(16, u32);
    var sum: u64 = 0;
    var i: usize = 0;

    while (i + 16 <= len) : (i += 16) {
        const va: Input = a[i..][0..16].*;
        const vb: Input = b[i..][0..16].*;
        const difference: Input = @max(va, vb) - @min(va, vb);
        const wide_difference: Wide = @intCast(difference);
        sum += @intCast(@reduce(.Add, wide_difference));
    }

    while (i < len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        sum += @intCast(if (x > y) x - y else y - x);
    }
    return sum;
}

pub export fn absdiff_u16x16(a: @Vector(16, u16), b: @Vector(16, u16)) @Vector(16, u16) {
    return @max(a, b) - @min(a, b);
}

pub export fn sad_u16x16(a: @Vector(16, u16), b: @Vector(16, u16)) u32 {
    const diff: @Vector(16, u16) = @max(a, b) - @min(a, b);
    const wide: @Vector(16, u32) = @intCast(diff);
    return @reduce(.Add, wide);
}
