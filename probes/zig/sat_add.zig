pub export fn sat_add_u8x32(a: @Vector(32, u8), b: @Vector(32, u8)) @Vector(32, u8) {
    return a +| b;
}

pub export fn sat_add_u8_scalar(dst: [*]u8, a: [*]const u8, b: [*]const u8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_i8_scalar(dst: [*]i8, a: [*]const i8, b: [*]const i8, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_u16_scalar(dst: [*]u16, a: [*]const u16, b: [*]const u16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_i16_scalar(dst: [*]i16, a: [*]const i16, b: [*]const i16, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_u32_scalar(dst: [*]u32, a: [*]const u32, b: [*]const u32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_i32_scalar(dst: [*]i32, a: [*]const i32, b: [*]const i32, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_u64_scalar(dst: [*]u64, a: [*]const u64, b: [*]const u64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}

pub export fn sat_add_i64_scalar(dst: [*]i64, a: [*]const i64, b: [*]const i64, len: usize) void {
    var i: usize = 0;
    while (i < len) : (i += 1) {
        dst[i] = a[i] +| b[i];
    }
}
