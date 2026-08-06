pub export fn sat_add_u8x32(a: @Vector(32, u8), b: @Vector(32, u8)) @Vector(32, u8) {
    return a +| b;
}

pub export fn sat_add_u8_scalar(dst: [*]u8, a: [*]const u8, b: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}
