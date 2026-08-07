const std = @import("std");
const kernels = @import("kernels.zig");

const sizes = [_]usize{ 1 << 10, 1 << 13, 1 << 16, 1 << 18, 1 << 20, 1 << 22 };
const warmup_samples: usize = 3;
const sample_count: usize = 15;
const target_elements_per_sample: usize = 1 << 20;
const max_iterations_per_sample: usize = 4096;
const half_values = [_]u16{ 0x0000, 0x3400, 0x3800, 0x3c00, 0x3e00, 0x4000, 0x4200, 0x4400 };

const Measurement = struct {
    iterations_per_sample: usize,
    ns_per_element: [sample_count]f64,
};

const Summary = struct {
    min: f64,
    median: f64,
    p95: f64,
    mad: f64,
};

fn iterationsFor(n: usize) usize {
    return @min(@max(target_elements_per_sample / n, 1), max_iterations_per_sample);
}

fn invoke(comptime function: anytype, args: anytype) void {
    // Both barriers matter for output-only kernels: their `void` result alone
    // cannot keep repeated stores alive or ordered across timed iterations.
    std.mem.doNotOptimizeAway(args);
    const result = @call(.auto, function, args);
    std.mem.doNotOptimizeAway(&result);
    std.mem.doNotOptimizeAway(args);
}

fn measure(io: std.Io, comptime function: anytype, args: anytype, n: usize) !Measurement {
    const iterations = iterationsFor(n);
    for (0..warmup_samples) |_| {
        for (0..iterations) |_| invoke(function, args);
    }

    var samples: [sample_count]f64 = undefined;
    for (&samples) |*sample| {
        const start = std.Io.Clock.now(.awake, io);
        for (0..iterations) |_| invoke(function, args);
        const end = std.Io.Clock.now(.awake, io);
        const elapsed: f64 = @floatFromInt(start.durationTo(end).toNanoseconds());
        sample.* = elapsed / @as(f64, @floatFromInt(iterations * n));
    }
    return .{ .iterations_per_sample = iterations, .ns_per_element = samples };
}

fn insertionSort(values: []f64) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const value = values[i];
        var position = i;
        while (position > 0 and values[position - 1] > value) : (position -= 1) {
            values[position] = values[position - 1];
        }
        values[position] = value;
    }
}

fn summarize(input: [sample_count]f64) Summary {
    var values = input;
    insertionSort(&values);
    const median = values[values.len / 2];
    var deviations: [sample_count]f64 = undefined;
    for (&deviations, values) |*deviation, value| {
        deviation.* = @abs(value - median);
    }
    insertionSort(&deviations);
    const p95_index = @min(values.len - 1, (values.len * 95 + 99) / 100 - 1);
    return .{
        .min = values[0],
        .median = median,
        .p95 = values[p95_index],
        .mad = deviations[deviations.len / 2],
    };
}

fn report(name: []const u8, n: usize, working_set_bytes: usize, effective_bytes_per_iteration: usize, measurement: Measurement) void {
    const summary = summarize(measurement.ns_per_element);
    const bytes_per_element = @as(f64, @floatFromInt(effective_bytes_per_iteration)) /
        @as(f64, @floatFromInt(n));
    const gib_s = bytes_per_element / (summary.median * 1e-9) /
        @as(f64, @floatFromInt(@as(u64, 1) << 30));
    std.debug.print(
        "RESULT name={s} n={d} working_set_bytes={d} effective_bytes_per_iteration={d} " ++
            "iterations_per_sample={d} sample_count={d} min_ns_per_element={d:.9} " ++
            "median_ns_per_element={d:.9} p95_ns_per_element={d:.9} mad_ns_per_element={d:.9} " ++
            "median_gib_per_second={d:.6} raw_ns_per_element=",
        .{ name, n, working_set_bytes, effective_bytes_per_iteration, measurement.iterations_per_sample, sample_count, summary.min, summary.median, summary.p95, summary.mad, gib_s },
    );
    for (measurement.ns_per_element, 0..) |value, index| {
        if (index != 0) std.debug.print(",", .{});
        std.debug.print("{d:.9}", .{value});
    }
    std.debug.print("\n", .{});
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    var threaded_io: std.Io.Threaded = .init_single_threaded;
    defer threaded_io.deinit();
    const io = threaded_io.io();

    for (sizes) |n| {
        {
            const x = try allocator.alloc(f32, n);
            defer allocator.free(x);
            const y = try allocator.alloc(f32, n);
            defer allocator.free(y);
            const dst = try allocator.alloc(f32, n);
            defer allocator.free(dst);

            for (x, y, 0..) |*xv, *yv, i| {
                xv.* = @as(f32, @floatFromInt(i)) * 0.001;
                yv.* = 1.0 + @as(f32, @floatFromInt(i)) * 0.0005;
            }

            var result = try measure(io, kernels.axpyScalar, .{ dst, x, y, @as(f32, 0.75) }, n);
            report("axpy/scalar-autovec", n, n * 12, n * 12, result);
            result = try measure(io, kernels.squaredErrorScalar, .{ x, y }, n);
            report("sqerr/scalar-f64", n, n * 8, n * 8, result);
            result = try measure(io, kernels.squaredErrorVector, .{ x, y }, n);
            report("sqerr/native-vector-f64", n, n * 8, n * 8, result);
            const dot_f32_scalar = kernels.dotF32Scalar(x, y);
            const dot_f32_vector = kernels.dotF32Vector(x, y);
            const dot_f32_error = @abs(dot_f32_scalar - dot_f32_vector) /
                @max(@abs(dot_f32_scalar), 1.0);
            std.debug.assert(dot_f32_error <= 1e-12);
            result = try measure(io, kernels.dotF32Scalar, .{ x, y }, n);
            report("dot-f32/scalar-f64", n, n * 8, n * 8, result);
            result = try measure(io, kernels.dotF32Vector, .{ x, y }, n);
            report("dot-f32/native-vector-f64", n, n * 8, n * 8, result);
        }

        {
            const a = try allocator.alloc(u8, n);
            defer allocator.free(a);
            const b = try allocator.alloc(u8, n);
            defer allocator.free(b);
            const sat_reference = try allocator.alloc(u8, n);
            defer allocator.free(sat_reference);
            const sat_dst = try allocator.alloc(u8, n);
            defer allocator.free(sat_dst);
            for (a, b, 0..) |*av, *bv, i| {
                av.* = @intCast((i * 17 + 3) & 255);
                bv.* = @intCast((i * 29 + 11) & 255);
            }
            std.debug.assert(kernels.sadU8Scalar(a, b) == kernels.sadU8Vector(a, b));
            kernels.satAddU8Scalar(sat_reference, a, b);
            kernels.satAddU8Vector(sat_dst, a, b);
            std.debug.assert(std.mem.eql(u8, sat_reference, sat_dst));

            var result = try measure(io, kernels.sadU8Scalar, .{ a, b }, n);
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, result);
            result = try measure(io, kernels.sadU8Vector, .{ a, b }, n);
            report("sad-u8/native-vector", n, n * 2, n * 2, result);

            result = try measure(io, kernels.satAddU8Scalar, .{ sat_dst, a, b }, n);
            report("sat-add-u8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satAddU8Vector, .{ sat_dst, a, b }, n);
            report("sat-add-u8/native-vector", n, n * 3, n * 3, result);
        }

        {
            const a = try allocator.alloc(f64, n);
            defer allocator.free(a);
            const b = try allocator.alloc(f64, n);
            defer allocator.free(b);
            for (a, b, 0..) |*av, *bv, i| {
                av.* = @as(f64, @floatFromInt(i)) * 0.001;
                bv.* = 1.0 + @as(f64, @floatFromInt(i)) * 0.0005;
            }

            const scalar = kernels.dotF64Scalar(a, b);
            const vector = kernels.dotF64Vector(a, b);
            const relative_error = @abs(scalar - vector) / @max(@abs(scalar), 1.0);
            std.debug.assert(relative_error <= 1e-12);

            var result = try measure(io, kernels.dotF64Scalar, .{ a, b }, n);
            report("dot-f64/scalar-f64", n, n * 16, n * 16, result);
            result = try measure(io, kernels.dotF64Vector, .{ a, b }, n);
            report("dot-f64/native-vector-f64", n, n * 16, n * 16, result);
        }

        {
            const i16_a = try allocator.alloc(i16, n);
            defer allocator.free(i16_a);
            const i16_b = try allocator.alloc(i16, n);
            defer allocator.free(i16_b);
            const u8_a = try allocator.alloc(u8, n);
            defer allocator.free(u8_a);
            const i8_a = try allocator.alloc(i8, n);
            defer allocator.free(i8_a);
            const i8_b = try allocator.alloc(i8, n);
            defer allocator.free(i8_b);
            const u8_u16_scalar = try allocator.alloc(u16, n);
            defer allocator.free(u8_u16_scalar);
            const u8_u16_vector = try allocator.alloc(u16, n);
            defer allocator.free(u8_u16_vector);
            const i8_i16_scalar = try allocator.alloc(i16, n);
            defer allocator.free(i8_i16_scalar);
            const i8_i16_vector = try allocator.alloc(i16, n);
            defer allocator.free(i8_i16_vector);

            const signed16 = [_]i16{ -32768, -32767, -1, 0, 1, 32767 };
            const signed8 = [_]i8{ -128, -127, -1, 0, 1, 127 };
            const unsigned8 = [_]u8{ 0, 1, 127, 128, 254, 255 };
            for (i16_a, i16_b, u8_a, i8_a, i8_b, 0..) |*ia, *ib, *ua, *sa, *sb, i| {
                ia.* = signed16[i % signed16.len];
                ib.* = signed16[(i * 5 + 1) % signed16.len];
                ua.* = unsigned8[i % unsigned8.len];
                sa.* = signed8[i % signed8.len];
                sb.* = signed8[(i * 5 + 1) % signed8.len];
            }

            const dot_i16_scalar = kernels.dotI16Scalar(i16_a, i16_b);
            const dot_i16_vector = kernels.dotI16Vector(i16_a, i16_b);
            std.debug.assert(dot_i16_scalar == dot_i16_vector);
            const dot_u8_i8_scalar = kernels.dotU8I8Scalar(u8_a, i8_b);
            const dot_u8_i8_vector = kernels.dotU8I8Vector(u8_a, i8_b);
            std.debug.assert(dot_u8_i8_scalar == dot_u8_i8_vector);

            kernels.widenMulU8U16Scalar(u8_u16_scalar, u8_a, u8_a);
            kernels.widenMulU8U16Vector(u8_u16_vector, u8_a, u8_a);
            std.debug.assert(std.mem.eql(u16, u8_u16_scalar, u8_u16_vector));
            kernels.widenMulI8I16Scalar(i8_i16_scalar, i8_a, i8_b);
            kernels.widenMulI8I16Vector(i8_i16_vector, i8_a, i8_b);
            std.debug.assert(std.mem.eql(i16, i8_i16_scalar, i8_i16_vector));

            var result = try measure(io, kernels.dotI16Scalar, .{ i16_a, i16_b }, n);
            report("dot-i16/scalar-i64", n, n * 4, n * 4, result);
            result = try measure(io, kernels.dotI16Vector, .{ i16_a, i16_b }, n);
            report("dot-i16/native-vector-i64", n, n * 4, n * 4, result);
            result = try measure(io, kernels.dotU8I8Scalar, .{ u8_a, i8_b }, n);
            report("dot-u8-i8/scalar-i64", n, n * 2, n * 2, result);
            result = try measure(io, kernels.dotU8I8Vector, .{ u8_a, i8_b }, n);
            report("dot-u8-i8/native-vector-i64", n, n * 2, n * 2, result);
            result = try measure(io, kernels.widenMulU8U16Scalar, .{ u8_u16_scalar, u8_a, u8_a }, n);
            report("widen-mul-u8-u16/scalar-autovec", n, n * 4, n * 4, result);
            result = try measure(io, kernels.widenMulU8U16Vector, .{ u8_u16_vector, u8_a, u8_a }, n);
            report("widen-mul-u8-u16/native-vector", n, n * 4, n * 4, result);
            result = try measure(io, kernels.widenMulI8I16Scalar, .{ i8_i16_scalar, i8_a, i8_b }, n);
            report("widen-mul-i8-i16/scalar-autovec", n, n * 4, n * 4, result);
            result = try measure(io, kernels.widenMulI8I16Vector, .{ i8_i16_vector, i8_a, i8_b }, n);
            report("widen-mul-i8-i16/native-vector", n, n * 4, n * 4, result);
        }

        {
            const u16_a = try allocator.alloc(u16, n);
            defer allocator.free(u16_a);
            const u16_b = try allocator.alloc(u16, n);
            defer allocator.free(u16_b);
            const i16_a = try allocator.alloc(i16, n);
            defer allocator.free(i16_a);
            const i16_b = try allocator.alloc(i16, n);
            defer allocator.free(i16_b);
            const u16_u32_scalar = try allocator.alloc(u32, n);
            defer allocator.free(u16_u32_scalar);
            const u16_u32_vector = try allocator.alloc(u32, n);
            defer allocator.free(u16_u32_vector);
            const i16_i32_scalar = try allocator.alloc(i32, n);
            defer allocator.free(i16_i32_scalar);
            const i16_i32_vector = try allocator.alloc(i32, n);
            defer allocator.free(i16_i32_vector);

            const unsigned16 = [_]u16{ 0, 1, 255, 256, 65534, 65535 };
            const signed16 = [_]i16{ -32768, -32767, -1, 0, 1, 32767 };
            for (u16_a, u16_b, i16_a, i16_b, 0..) |*ua, *ub, *ia, *ib, i| {
                ua.* = unsigned16[i % unsigned16.len];
                ub.* = unsigned16[(i * 5 + 1) % unsigned16.len];
                ia.* = signed16[i % signed16.len];
                ib.* = signed16[(i * 5 + 1) % signed16.len];
            }

            kernels.widenMulU16U32Scalar(u16_u32_scalar, u16_a, u16_b);
            kernels.widenMulU16U32Vector(u16_u32_vector, u16_a, u16_b);
            std.debug.assert(std.mem.eql(u32, u16_u32_scalar, u16_u32_vector));
            kernels.widenMulI16I32Scalar(i16_i32_scalar, i16_a, i16_b);
            kernels.widenMulI16I32Vector(i16_i32_vector, i16_a, i16_b);
            std.debug.assert(std.mem.eql(i32, i16_i32_scalar, i16_i32_vector));

            var result = try measure(io, kernels.widenMulU16U32Scalar, .{ u16_u32_scalar, u16_a, u16_b }, n);
            report("widen-mul-u16-u32/scalar-autovec", n, n * 8, n * 8, result);
            result = try measure(io, kernels.widenMulU16U32Vector, .{ u16_u32_vector, u16_a, u16_b }, n);
            report("widen-mul-u16-u32/native-vector", n, n * 8, n * 8, result);
            result = try measure(io, kernels.widenMulI16I32Scalar, .{ i16_i32_scalar, i16_a, i16_b }, n);
            report("widen-mul-i16-i32/scalar-autovec", n, n * 8, n * 8, result);
            result = try measure(io, kernels.widenMulI16I32Vector, .{ i16_i32_vector, i16_a, i16_b }, n);
            report("widen-mul-i16-i32/native-vector", n, n * 8, n * 8, result);
        }

        {
            const c = try allocator.alloc(f16, n);
            defer allocator.free(c);
            const lo = try allocator.alloc(f16, n);
            defer allocator.free(lo);
            const hi = try allocator.alloc(f16, n);
            defer allocator.free(hi);
            const native = try allocator.alloc(f16, n);
            defer allocator.free(native);
            const promoted = try allocator.alloc(f16, n);
            defer allocator.free(promoted);

            for (c, lo, hi, 0..) |*value, *low, *high, i| {
                value.* = @bitCast(half_values[i & 7]);
                low.* = @bitCast(@as(u16, 0x3800));
                high.* = @bitCast(@as(u16, 0x4000));
            }
            kernels.clampF16Native(native, c, lo, hi);
            kernels.clampF16PromoteOnce(promoted, c, lo, hi);
            std.debug.assert(std.mem.eql(f16, native, promoted));

            var result = try measure(io, kernels.clampF16Native, .{ native, c, lo, hi }, n);
            report("clamp-f16/native", n, n * 8, n * 8, result);
            result = try measure(io, kernels.clampF16PromoteOnce, .{ promoted, c, lo, hi }, n);
            report("clamp-f16/promote-f32", n, n * 8, n * 8, result);
        }
    }

    std.debug.print(
        "META size_count={d} warmup_samples={d} sample_count={d} " ++
            "target_elements_per_sample={d} sat_add_u8_dispatch_tier=native-vector\n",
        .{ sizes.len, warmup_samples, sample_count, target_elements_per_sample },
    );
}
