// u8 absolute-difference and widening-reduction probes.

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
