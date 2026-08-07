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
            const sat_sub_reference = try allocator.alloc(u8, n);
            defer allocator.free(sat_sub_reference);
            const sat_sub_dst = try allocator.alloc(u8, n);
            defer allocator.free(sat_sub_dst);
            for (a, b, 0..) |*av, *bv, i| {
                av.* = @intCast((i * 17 + 3) & 255);
                bv.* = @intCast((i * 29 + 11) & 255);
            }
            std.debug.assert(kernels.sadU8Scalar(a, b) == kernels.sadU8Vector(a, b));
            kernels.satAddU8Scalar(sat_reference, a, b);
            kernels.satAddU8Vector(sat_dst, a, b);
            std.debug.assert(std.mem.eql(u8, sat_reference, sat_dst));
            kernels.satSubU8Scalar(sat_sub_reference, a, b);
            kernels.satSubU8Vector(sat_sub_dst, a, b);
            std.debug.assert(std.mem.eql(u8, sat_sub_reference, sat_sub_dst));

            var result = try measure(io, kernels.sadU8Scalar, .{ a, b }, n);
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, result);
            result = try measure(io, kernels.sadU8Vector, .{ a, b }, n);
            report("sad-u8/native-vector", n, n * 2, n * 2, result);

            result = try measure(io, kernels.satAddU8Scalar, .{ sat_dst, a, b }, n);
            report("sat-add-u8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satAddU8Vector, .{ sat_dst, a, b }, n);
            report("sat-add-u8/native-vector", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satSubU8Scalar, .{ sat_sub_dst, a, b }, n);
            report("sat-sub-u8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satSubU8Vector, .{ sat_sub_dst, a, b }, n);
            report("sat-sub-u8/native-vector", n, n * 3, n * 3, result);
        }

        {
            const u16_a = try allocator.alloc(u16, n);
            defer allocator.free(u16_a);
            const u16_b = try allocator.alloc(u16, n);
            defer allocator.free(u16_b);
            for (u16_a, u16_b, 0..) |*av, *bv, i| {
                av.* = @truncate(i * 257 + 3);
                bv.* = @truncate(i * 509 + 11);
            }
            std.debug.assert(kernels.sadU16Scalar(u16_a, u16_b) == kernels.sadU16Vector(u16_a, u16_b));

            var result = try measure(io, kernels.sadU16Scalar, .{ u16_a, u16_b }, n);
            report("sad-u16/scalar-autovec", n, n * 4, n * 4, result);
            result = try measure(io, kernels.sadU16Vector, .{ u16_a, u16_b }, n);
            report("sad-u16/native-vector", n, n * 4, n * 4, result);
        }


        {
            const i8_a = try allocator.alloc(i8, n);
            defer allocator.free(i8_a);
            const i8_b = try allocator.alloc(i8, n);
            defer allocator.free(i8_b);
            const i8_reference = try allocator.alloc(i8, n);
            defer allocator.free(i8_reference);
            const i8_dst = try allocator.alloc(i8, n);
            defer allocator.free(i8_dst);
            const u16_a = try allocator.alloc(u16, n);
            defer allocator.free(u16_a);
            const u16_b = try allocator.alloc(u16, n);
            defer allocator.free(u16_b);
            const u16_reference = try allocator.alloc(u16, n);
            defer allocator.free(u16_reference);
            const u16_dst = try allocator.alloc(u16, n);
            defer allocator.free(u16_dst);
            const i16_a = try allocator.alloc(i16, n);
            defer allocator.free(i16_a);
            const i16_b = try allocator.alloc(i16, n);
            defer allocator.free(i16_b);
            const i16_reference = try allocator.alloc(i16, n);
            defer allocator.free(i16_reference);
            const i16_dst = try allocator.alloc(i16, n);
            defer allocator.free(i16_dst);
            const u32_a = try allocator.alloc(u32, n);
            defer allocator.free(u32_a);
            const u32_b = try allocator.alloc(u32, n);
            defer allocator.free(u32_b);
            const u32_reference = try allocator.alloc(u32, n);
            defer allocator.free(u32_reference);
            const u32_dst = try allocator.alloc(u32, n);
            defer allocator.free(u32_dst);
            const i32_a = try allocator.alloc(i32, n);
            defer allocator.free(i32_a);
            const i32_b = try allocator.alloc(i32, n);
            defer allocator.free(i32_b);
            const i32_reference = try allocator.alloc(i32, n);
            defer allocator.free(i32_reference);
            const i32_dst = try allocator.alloc(i32, n);
            defer allocator.free(i32_dst);
            const u64_a = try allocator.alloc(u64, n);
            defer allocator.free(u64_a);
            const u64_b = try allocator.alloc(u64, n);
            defer allocator.free(u64_b);
            const u64_reference = try allocator.alloc(u64, n);
            defer allocator.free(u64_reference);
            const u64_dst = try allocator.alloc(u64, n);
            defer allocator.free(u64_dst);
            const i64_a = try allocator.alloc(i64, n);
            defer allocator.free(i64_a);
            const i64_b = try allocator.alloc(i64, n);
            defer allocator.free(i64_b);
            const i64_reference = try allocator.alloc(i64, n);
            defer allocator.free(i64_reference);
            const i64_dst = try allocator.alloc(i64, n);
            defer allocator.free(i64_dst);

            const signed8 = [_]i8{ -128, -127, -1, 0, 1, 126, 127 };
            const unsigned16 = [_]u16{ 0, 1, 32767, 32768, 65534, 65535, 65535 };
            const signed16 = [_]i16{ -32768, -32767, -1, 0, 1, 32766, 32767 };
            const unsigned32 = [_]u32{ 0, 1, 2147483647, 2147483648, 4294967294, 4294967295, 4294967295 };
            const signed32 = [_]i32{ -2147483648, -2147483647, -1, 0, 1, 2147483646, 2147483647 };
            const unsigned64 = [_]u64{
                0,
                1,
                0x7fff_ffff_ffff_ffff,
                0x8000_0000_0000_0000,
                std.math.maxInt(u64) - 1,
                std.math.maxInt(u64),
                std.math.maxInt(u64),
            };
            const signed64 = [_]i64{
                std.math.minInt(i64),
                std.math.minInt(i64) + 1,
                -1,
                0,
                1,
                std.math.maxInt(i64) - 1,
                std.math.maxInt(i64),
            };
            const pair_indices = [_]usize{ 1, 0, 6, 2, 3, 5, 6 };
            for (i8_a, i8_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = signed8[lane];
                b.* = signed8[pair_indices[lane]];
            }
            for (u16_a, u16_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = unsigned16[lane];
                b.* = unsigned16[pair_indices[lane]];
            }
            for (i16_a, i16_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = signed16[lane];
                b.* = signed16[pair_indices[lane]];
            }
            for (u32_a, u32_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = unsigned32[lane];
                b.* = unsigned32[pair_indices[lane]];
            }
            for (i32_a, i32_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = signed32[lane];
                b.* = signed32[pair_indices[lane]];
            }
            for (u64_a, u64_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = unsigned64[lane];
                b.* = unsigned64[pair_indices[lane]];
            }
            for (i64_a, i64_b, 0..) |*a, *b, i| {
                const lane = i % pair_indices.len;
                a.* = signed64[lane];
                b.* = signed64[pair_indices[lane]];
            }

            for (i8_reference, i8_a, i8_b) |*out, a, b| {
                const sum: i16 = @as(i16, a) + @as(i16, b);
                const min_value: i16 = @intCast(std.math.minInt(i8));
                const max_value: i16 = @intCast(std.math.maxInt(i8));
                out.* = if (sum < min_value) std.math.minInt(i8) else if (sum > max_value) std.math.maxInt(i8) else @intCast(sum);
            }
            for (u16_reference, u16_a, u16_b) |*out, a, b| {
                const sum: u32 = @as(u32, a) + @as(u32, b);
                out.* = if (sum > @as(u32, std.math.maxInt(u16))) std.math.maxInt(u16) else @intCast(sum);
            }
            for (i16_reference, i16_a, i16_b) |*out, a, b| {
                const sum: i32 = @as(i32, a) + @as(i32, b);
                const min_value: i32 = @intCast(std.math.minInt(i16));
                const max_value: i32 = @intCast(std.math.maxInt(i16));
                out.* = if (sum < min_value) std.math.minInt(i16) else if (sum > max_value) std.math.maxInt(i16) else @intCast(sum);
            }
            for (u32_reference, u32_a, u32_b) |*out, a, b| {
                const sum: u64 = @as(u64, a) + @as(u64, b);
                out.* = if (sum > @as(u64, std.math.maxInt(u32))) std.math.maxInt(u32) else @intCast(sum);
            }
            for (i32_reference, i32_a, i32_b) |*out, a, b| {
                const sum: i64 = @as(i64, a) + @as(i64, b);
                const min_value: i64 = @intCast(std.math.minInt(i32));
                const max_value: i64 = @intCast(std.math.maxInt(i32));
                out.* = if (sum < min_value) std.math.minInt(i32) else if (sum > max_value) std.math.maxInt(i32) else @intCast(sum);
            }
            for (u64_reference, u64_a, u64_b) |*out, a, b| {
                const sum: u128 = @as(u128, a) + @as(u128, b);
                out.* = if (sum > @as(u128, std.math.maxInt(u64))) std.math.maxInt(u64) else @intCast(sum);
            }
            for (i64_reference, i64_a, i64_b) |*out, a, b| {
                const sum: i128 = @as(i128, a) + @as(i128, b);
                const min_value: i128 = @intCast(std.math.minInt(i64));
                const max_value: i128 = @intCast(std.math.maxInt(i64));
                out.* = if (sum < min_value) std.math.minInt(i64) else if (sum > max_value) std.math.maxInt(i64) else @intCast(sum);
            }

            kernels.satAddI8Scalar(i8_dst, i8_a, i8_b);
            std.debug.assert(std.mem.eql(i8, i8_reference, i8_dst));
            kernels.satAddU16Scalar(u16_dst, u16_a, u16_b);
            std.debug.assert(std.mem.eql(u16, u16_reference, u16_dst));
            kernels.satAddI16Scalar(i16_dst, i16_a, i16_b);
            std.debug.assert(std.mem.eql(i16, i16_reference, i16_dst));
            kernels.satAddU32Scalar(u32_dst, u32_a, u32_b);
            std.debug.assert(std.mem.eql(u32, u32_reference, u32_dst));
            kernels.satAddI32Scalar(i32_dst, i32_a, i32_b);
            std.debug.assert(std.mem.eql(i32, i32_reference, i32_dst));
            kernels.satAddU64Scalar(u64_dst, u64_a, u64_b);
            std.debug.assert(std.mem.eql(u64, u64_reference, u64_dst));
            kernels.satAddI64Scalar(i64_dst, i64_a, i64_b);
            std.debug.assert(std.mem.eql(i64, i64_reference, i64_dst));
            for (i8_reference, i8_a, i8_b) |*out, a, b| {
                const difference: i16 = @as(i16, a) - @as(i16, b);
                const min_value: i16 = @intCast(std.math.minInt(i8));
                const max_value: i16 = @intCast(std.math.maxInt(i8));
                out.* = if (difference < min_value) std.math.minInt(i8) else if (difference > max_value) std.math.maxInt(i8) else @intCast(difference);
            }
            for (u16_reference, u16_a, u16_b) |*out, a, b| {
                const difference: i32 = @as(i32, a) - @as(i32, b);
                out.* = if (difference < 0) 0 else @intCast(difference);
            }
            for (i16_reference, i16_a, i16_b) |*out, a, b| {
                const difference: i32 = @as(i32, a) - @as(i32, b);
                const min_value: i32 = @intCast(std.math.minInt(i16));
                const max_value: i32 = @intCast(std.math.maxInt(i16));
                out.* = if (difference < min_value) std.math.minInt(i16) else if (difference > max_value) std.math.maxInt(i16) else @intCast(difference);
            }
            for (u32_reference, u32_a, u32_b) |*out, a, b| {
                const difference: i64 = @as(i64, a) - @as(i64, b);
                out.* = if (difference < 0) 0 else @intCast(difference);
            }
            for (i32_reference, i32_a, i32_b) |*out, a, b| {
                const difference: i64 = @as(i64, a) - @as(i64, b);
                const min_value: i64 = @intCast(std.math.minInt(i32));
                const max_value: i64 = @intCast(std.math.maxInt(i32));
                out.* = if (difference < min_value) std.math.minInt(i32) else if (difference > max_value) std.math.maxInt(i32) else @intCast(difference);
            }
            for (u64_reference, u64_a, u64_b) |*out, a, b| {
                const difference: i128 = @as(i128, a) - @as(i128, b);
                out.* = if (difference < 0) 0 else @intCast(difference);
            }
            for (i64_reference, i64_a, i64_b) |*out, a, b| {
                const difference: i128 = @as(i128, a) - @as(i128, b);
                const min_value: i128 = @intCast(std.math.minInt(i64));
                const max_value: i128 = @intCast(std.math.maxInt(i64));
                out.* = if (difference < min_value) std.math.minInt(i64) else if (difference > max_value) std.math.maxInt(i64) else @intCast(difference);
            }
            kernels.satSubI8Scalar(i8_dst, i8_a, i8_b);
            std.debug.assert(std.mem.eql(i8, i8_reference, i8_dst));
            kernels.satSubU16Scalar(u16_dst, u16_a, u16_b);
            std.debug.assert(std.mem.eql(u16, u16_reference, u16_dst));
            kernels.satSubI16Scalar(i16_dst, i16_a, i16_b);
            std.debug.assert(std.mem.eql(i16, i16_reference, i16_dst));
            kernels.satSubU32Scalar(u32_dst, u32_a, u32_b);
            std.debug.assert(std.mem.eql(u32, u32_reference, u32_dst));
            kernels.satSubI32Scalar(i32_dst, i32_a, i32_b);
            std.debug.assert(std.mem.eql(i32, i32_reference, i32_dst));
            kernels.satSubU64Scalar(u64_dst, u64_a, u64_b);
            std.debug.assert(std.mem.eql(u64, u64_reference, u64_dst));
            kernels.satSubI64Scalar(i64_dst, i64_a, i64_b);
            std.debug.assert(std.mem.eql(i64, i64_reference, i64_dst));

            var result = try measure(io, kernels.satAddI8Scalar, .{ i8_dst, i8_a, i8_b }, n);
            report("sat-add-i8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satAddU16Scalar, .{ u16_dst, u16_a, u16_b }, n);
            report("sat-add-u16/scalar-autovec", n, n * 6, n * 6, result);
            result = try measure(io, kernels.satAddI16Scalar, .{ i16_dst, i16_a, i16_b }, n);
            report("sat-add-i16/scalar-autovec", n, n * 6, n * 6, result);
            result = try measure(io, kernels.satAddU32Scalar, .{ u32_dst, u32_a, u32_b }, n);
            report("sat-add-u32/scalar-autovec", n, n * 12, n * 12, result);
            result = try measure(io, kernels.satAddI32Scalar, .{ i32_dst, i32_a, i32_b }, n);
            report("sat-add-i32/scalar-autovec", n, n * 12, n * 12, result);
            result = try measure(io, kernels.satAddU64Scalar, .{ u64_dst, u64_a, u64_b }, n);
            report("sat-add-u64/scalar-autovec", n, n * 24, n * 24, result);
            result = try measure(io, kernels.satAddI64Scalar, .{ i64_dst, i64_a, i64_b }, n);
            report("sat-add-i64/scalar-autovec", n, n * 24, n * 24, result);
            result = try measure(io, kernels.satSubI8Scalar, .{ i8_dst, i8_a, i8_b }, n);
            report("sat-sub-i8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.satSubU16Scalar, .{ u16_dst, u16_a, u16_b }, n);
            report("sat-sub-u16/scalar-autovec", n, n * 6, n * 6, result);
            result = try measure(io, kernels.satSubI16Scalar, .{ i16_dst, i16_a, i16_b }, n);
            report("sat-sub-i16/scalar-autovec", n, n * 6, n * 6, result);
            result = try measure(io, kernels.satSubU32Scalar, .{ u32_dst, u32_a, u32_b }, n);
            report("sat-sub-u32/scalar-autovec", n, n * 12, n * 12, result);
            result = try measure(io, kernels.satSubI32Scalar, .{ i32_dst, i32_a, i32_b }, n);
            report("sat-sub-i32/scalar-autovec", n, n * 12, n * 12, result);
            result = try measure(io, kernels.satSubU64Scalar, .{ u64_dst, u64_a, u64_b }, n);
            report("sat-sub-u64/scalar-autovec", n, n * 24, n * 24, result);
            result = try measure(io, kernels.satSubI64Scalar, .{ i64_dst, i64_a, i64_b }, n);
            report("sat-sub-i64/scalar-autovec", n, n * 24, n * 24, result);
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
            const u8_b = try allocator.alloc(u8, n);
            defer allocator.free(u8_b);
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
            for (i16_a, i16_b, u8_a, u8_b, i8_a, i8_b, 0..) |*ia, *ib, *ua, *ub, *sa, *sb, i| {
                ia.* = signed16[i % signed16.len];
                ib.* = signed16[(i * 5 + 1) % signed16.len];
                ua.* = unsigned8[i % unsigned8.len];
                ub.* = unsigned8[(i * 5 + 1) % unsigned8.len];
                sa.* = signed8[i % signed8.len];
                sb.* = signed8[(i * 5 + 1) % signed8.len];
            }

            const dot_i16_scalar = kernels.dotI16Scalar(i16_a, i16_b);
            const dot_i16_vector = kernels.dotI16Vector(i16_a, i16_b);
            std.debug.assert(dot_i16_scalar == dot_i16_vector);
            const dot_u8_i8_scalar = kernels.dotU8I8Scalar(u8_a, i8_b);
            const dot_u8_i8_vector = kernels.dotU8I8Vector(u8_a, i8_b);
            std.debug.assert(dot_u8_i8_scalar == dot_u8_i8_vector);

            kernels.widenMulU8U16Scalar(u8_u16_scalar, u8_a, u8_b);
            kernels.widenMulU8U16Vector(u8_u16_vector, u8_a, u8_b);
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
            result = try measure(io, kernels.widenMulU8U16Scalar, .{ u8_u16_scalar, u8_a, u8_b }, n);
            report("widen-mul-u8-u16/scalar-autovec", n, n * 4, n * 4, result);
            result = try measure(io, kernels.widenMulU8U16Vector, .{ u8_u16_vector, u8_a, u8_b }, n);
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
            const u32_a = try allocator.alloc(u32, n);
            defer allocator.free(u32_a);
            const u32_b = try allocator.alloc(u32, n);
            defer allocator.free(u32_b);
            const i32_a = try allocator.alloc(i32, n);
            defer allocator.free(i32_a);
            const i32_b = try allocator.alloc(i32, n);
            defer allocator.free(i32_b);
            const u32_u64_scalar = try allocator.alloc(u64, n);
            defer allocator.free(u32_u64_scalar);
            const u32_u64_vector = try allocator.alloc(u64, n);
            defer allocator.free(u32_u64_vector);
            const i32_i64_scalar = try allocator.alloc(i64, n);
            defer allocator.free(i32_i64_scalar);
            const i32_i64_vector = try allocator.alloc(i64, n);
            defer allocator.free(i32_i64_vector);

            const unsigned32 = [_]u32{ 0, 1, 255, 256, 4294967294, 4294967295 };
            const signed32 = [_]i32{ -2147483648, -2147483647, -1, 0, 1, 2147483647 };
            for (u32_a, u32_b, i32_a, i32_b, 0..) |*ua, *ub, *ia, *ib, i| {
                ua.* = unsigned32[i % unsigned32.len];
                ub.* = unsigned32[(i * 5 + 1) % unsigned32.len];
                ia.* = signed32[i % signed32.len];
                ib.* = signed32[(i * 5 + 1) % signed32.len];
            }

            kernels.widenMulU32U64Scalar(u32_u64_scalar, u32_a, u32_b);
            kernels.widenMulU32U64Vector(u32_u64_vector, u32_a, u32_b);
            std.debug.assert(std.mem.eql(u64, u32_u64_scalar, u32_u64_vector));
            kernels.widenMulI32I64Scalar(i32_i64_scalar, i32_a, i32_b);
            kernels.widenMulI32I64Vector(i32_i64_vector, i32_a, i32_b);
            std.debug.assert(std.mem.eql(i64, i32_i64_scalar, i32_i64_vector));

            var result = try measure(io, kernels.widenMulU32U64Scalar, .{ u32_u64_scalar, u32_a, u32_b }, n);
            report("widen-mul-u32-u64/scalar-autovec", n, n * 16, n * 16, result);
            result = try measure(io, kernels.widenMulU32U64Vector, .{ u32_u64_vector, u32_a, u32_b }, n);
            report("widen-mul-u32-u64/native-vector", n, n * 16, n * 16, result);
            result = try measure(io, kernels.widenMulI32I64Scalar, .{ i32_i64_scalar, i32_a, i32_b }, n);
            report("widen-mul-i32-i64/scalar-autovec", n, n * 16, n * 16, result);
            result = try measure(io, kernels.widenMulI32I64Vector, .{ i32_i64_vector, i32_a, i32_b }, n);
            report("widen-mul-i32-i64/native-vector", n, n * 16, n * 16, result);
        }

        {
            const u8_src = try allocator.alloc(u8, n);
            defer allocator.free(u8_src);
            const i8_src = try allocator.alloc(i8, n);
            defer allocator.free(i8_src);
            const u16_src = try allocator.alloc(u16, n);
            defer allocator.free(u16_src);
            const i16_src = try allocator.alloc(i16, n);
            defer allocator.free(i16_src);
            const u8_u16_scalar = try allocator.alloc(u16, n);
            defer allocator.free(u8_u16_scalar);
            const u8_u16_vector = try allocator.alloc(u16, n);
            defer allocator.free(u8_u16_vector);
            const u8_u32_scalar = try allocator.alloc(u32, n);
            defer allocator.free(u8_u32_scalar);
            const u8_u32_vector = try allocator.alloc(u32, n);
            defer allocator.free(u8_u32_vector);
            const i8_i16_scalar = try allocator.alloc(i16, n);
            defer allocator.free(i8_i16_scalar);
            const i8_i16_vector = try allocator.alloc(i16, n);
            defer allocator.free(i8_i16_vector);
            const u16_u32_scalar = try allocator.alloc(u32, n);
            defer allocator.free(u16_u32_scalar);
            const u16_u32_vector = try allocator.alloc(u32, n);
            defer allocator.free(u16_u32_vector);
            const i16_i32_scalar = try allocator.alloc(i32, n);
            defer allocator.free(i16_i32_scalar);
            const i16_i32_vector = try allocator.alloc(i32, n);
            defer allocator.free(i16_i32_vector);
            const u16_f32_scalar = try allocator.alloc(f32, n);
            defer allocator.free(u16_f32_scalar);
            const u16_f32_vector = try allocator.alloc(f32, n);
            defer allocator.free(u16_f32_vector);
            const i16_f32_scalar = try allocator.alloc(f32, n);
            defer allocator.free(i16_f32_scalar);
            const i16_f32_vector = try allocator.alloc(f32, n);
            defer allocator.free(i16_f32_vector);

            for (u8_src, i8_src, u16_src, i16_src, 0..) |*u8_value, *i8_value, *u16_value, *i16_value, i| {
                const byte: u8 = @truncate(i * 37 + 11);
                u8_value.* = byte;
                i8_value.* = @bitCast(byte);
                u16_value.* = @truncate(i * 1021 + 17);
                i16_value.* = @bitCast(@as(u16, @truncate(i * 1973 + 29)));
            }

            kernels.widenU8ToU16Scalar(u8_u16_scalar, u8_src);
            kernels.widenU8ToU16Vector(u8_u16_vector, u8_src);
            kernels.widenU8ToU32Scalar(u8_u32_scalar, u8_src);
            kernels.widenU8ToU32Vector(u8_u32_vector, u8_src);
            kernels.widenI8ToI16Scalar(i8_i16_scalar, i8_src);
            kernels.widenI8ToI16Vector(i8_i16_vector, i8_src);
            kernels.widenU16ToU32Scalar(u16_u32_scalar, u16_src);
            kernels.widenU16ToU32Vector(u16_u32_vector, u16_src);
            kernels.widenI16ToI32Scalar(i16_i32_scalar, i16_src);
            kernels.widenI16ToI32Vector(i16_i32_vector, i16_src);
            kernels.convertU16ToF32Scalar(u16_f32_scalar, u16_src);
            kernels.convertU16ToF32Vector(u16_f32_vector, u16_src);
            kernels.convertI16ToF32Scalar(i16_f32_scalar, i16_src);
            kernels.convertI16ToF32Vector(i16_f32_vector, i16_src);

            std.debug.assert(std.mem.eql(u16, u8_u16_scalar, u8_u16_vector));
            std.debug.assert(std.mem.eql(u32, u8_u32_scalar, u8_u32_vector));
            std.debug.assert(std.mem.eql(i16, i8_i16_scalar, i8_i16_vector));
            std.debug.assert(std.mem.eql(u32, u16_u32_scalar, u16_u32_vector));
            std.debug.assert(std.mem.eql(i32, i16_i32_scalar, i16_i32_vector));
            std.debug.assert(std.mem.eql(f32, u16_f32_scalar, u16_f32_vector));
            std.debug.assert(std.mem.eql(f32, i16_f32_scalar, i16_f32_vector));
            for (u8_u16_scalar, u8_src, u8_u32_scalar) |u16_value, u8_value, u32_value| {
                std.debug.assert(u16_value == @as(u16, u8_value));
                std.debug.assert(u32_value == @as(u32, u8_value));
            }
            for (i8_i16_scalar, i8_src) |wide_value, value| {
                std.debug.assert(wide_value == @as(i16, value));
            }
            for (u16_u32_scalar, u16_src, u16_f32_scalar) |u32_value, value, float_value| {
                std.debug.assert(u32_value == @as(u32, value));
                std.debug.assert(float_value == @as(f32, @floatFromInt(value)));
            }
            for (i16_i32_scalar, i16_src, i16_f32_scalar) |i32_value, value, float_value| {
                std.debug.assert(i32_value == @as(i32, value));
                std.debug.assert(float_value == @as(f32, @floatFromInt(value)));
            }

            var result = try measure(io, kernels.widenU8ToU16Scalar, .{ u8_u16_scalar, u8_src }, n);
            report("widen-u8-u16/scalar", n, n * 3, n * 3, result);
            result = try measure(io, kernels.widenU8ToU16Vector, .{ u8_u16_vector, u8_src }, n);
            report("widen-u8-u16/native-vector", n, n * 3, n * 3, result);
            result = try measure(io, kernels.widenU8ToU32Scalar, .{ u8_u32_scalar, u8_src }, n);
            report("widen-u8-u32/scalar", n, n * 5, n * 5, result);
            result = try measure(io, kernels.widenU8ToU32Vector, .{ u8_u32_vector, u8_src }, n);
            report("widen-u8-u32/native-vector", n, n * 5, n * 5, result);
            result = try measure(io, kernels.widenI8ToI16Scalar, .{ i8_i16_scalar, i8_src }, n);
            report("widen-i8-i16/scalar", n, n * 3, n * 3, result);
            result = try measure(io, kernels.widenI8ToI16Vector, .{ i8_i16_vector, i8_src }, n);
            report("widen-i8-i16/native-vector", n, n * 3, n * 3, result);
            result = try measure(io, kernels.widenU16ToU32Scalar, .{ u16_u32_scalar, u16_src }, n);
            report("widen-u16-u32/scalar", n, n * 6, n * 6, result);
            result = try measure(io, kernels.widenU16ToU32Vector, .{ u16_u32_vector, u16_src }, n);
            report("widen-u16-u32/native-vector", n, n * 6, n * 6, result);
            result = try measure(io, kernels.widenI16ToI32Scalar, .{ i16_i32_scalar, i16_src }, n);
            report("widen-i16-i32/scalar", n, n * 6, n * 6, result);
            result = try measure(io, kernels.widenI16ToI32Vector, .{ i16_i32_vector, i16_src }, n);
            report("widen-i16-i32/native-vector", n, n * 6, n * 6, result);
            result = try measure(io, kernels.convertU16ToF32Scalar, .{ u16_f32_scalar, u16_src }, n);
            report("convert-u16-f32/scalar", n, n * 6, n * 6, result);
            result = try measure(io, kernels.convertU16ToF32Vector, .{ u16_f32_vector, u16_src }, n);
            report("convert-u16-f32/native-vector", n, n * 6, n * 6, result);
            result = try measure(io, kernels.convertI16ToF32Scalar, .{ i16_f32_scalar, i16_src }, n);
            report("convert-i16-f32/scalar", n, n * 6, n * 6, result);
            result = try measure(io, kernels.convertI16ToF32Vector, .{ i16_f32_vector, i16_src }, n);
            report("convert-i16-f32/native-vector", n, n * 6, n * 6, result);
        }

        {
            const u8_src = try allocator.alloc(u8, n);
            defer allocator.free(u8_src);
            const u16_src = try allocator.alloc(u16, n);
            defer allocator.free(u16_src);
            const f32_src = try allocator.alloc(f32, n);
            defer allocator.free(f32_src);
            const affine_dst = try allocator.alloc(f32, n);
            defer allocator.free(affine_dst);
            const f32_u8_trunc = try allocator.alloc(u8, n);
            defer allocator.free(f32_u8_trunc);
            const f32_u8_round = try allocator.alloc(u8, n);
            defer allocator.free(f32_u8_round);
            const f32_u8_sat = try allocator.alloc(u8, n);
            defer allocator.free(f32_u8_sat);
            const f32_u16_sat = try allocator.alloc(u16, n);
            defer allocator.free(f32_u16_sat);
            const narrow_trunc = try allocator.alloc(u8, n);
            defer allocator.free(narrow_trunc);
            const narrow_round = try allocator.alloc(u8, n);
            defer allocator.free(narrow_round);
            const narrow_sat = try allocator.alloc(u8, n);
            defer allocator.free(narrow_sat);

            for (u8_src, u16_src, f32_src, 0..) |*u8_value, *u16_value, *float_value, i| {
                u8_value.* = @truncate(i * 29 + 7);
                u16_value.* = @truncate(i * 997 + 13);
                const integer_part: u32 = @intCast(i % 255);
                float_value.* = @as(f32, @floatFromInt(integer_part)) + 0.25;
            }
            kernels.convertU8F32AffineScalar(affine_dst, u8_src, 1.25, -2.5);
            kernels.convertF32U8TruncScalar(f32_u8_trunc, f32_src);
            kernels.convertF32U8RoundScalar(f32_u8_round, f32_src);
            kernels.convertF32U8SatScalar(f32_u8_sat, f32_src);
            kernels.f32ToU16SatScalar(f32_u16_sat, f32_src);
            kernels.narrowU16ToU8TruncScalar(narrow_trunc, u16_src);
            kernels.narrowU16ToU8RoundScalar(narrow_round, u16_src);
            kernels.narrowU16ToU8SatScalar(narrow_sat, u16_src);

            for (affine_dst, u8_src) |value, source| {
                std.debug.assert(value == @as(f32, @floatFromInt(source)) * 1.25 - 2.5);
            }
            for (f32_u8_trunc, f32_u8_round, f32_u8_sat, f32_u16_sat, f32_src) |trunc_value, round_value, sat_value, u16_value, value| {
                const expected_trunc: u8 = @intFromFloat(value);
                const expected_round: u8 = @intFromFloat(@floor(value + 0.5));
                std.debug.assert(trunc_value == expected_trunc);
                std.debug.assert(round_value == expected_round);
                std.debug.assert(sat_value == expected_trunc);
                std.debug.assert(u16_value == @as(u16, @intFromFloat(value)));
            }
            for (narrow_trunc, narrow_round, narrow_sat, u16_src) |trunc_value, round_value, sat_value, value| {
                std.debug.assert(trunc_value == @as(u8, @intCast(value & 0xff)));
                std.debug.assert(round_value == @as(u8, @intCast((@as(u32, value) + 128) / 257)));
                std.debug.assert(sat_value == if (value > 255) @as(u8, 255) else @as(u8, @intCast(value)));
            }

            var result = try measure(io, kernels.convertU8F32AffineScalar, .{ affine_dst, u8_src, @as(f32, 1.25), @as(f32, -2.5) }, n);
            report("convert-u8-f32-affine/scalar", n, n * 5, n * 5, result);
            result = try measure(io, kernels.convertF32U8TruncScalar, .{ f32_u8_trunc, f32_src }, n);
            report("convert-f32-u8-trunc/scalar", n, n * 5, n * 5, result);
            result = try measure(io, kernels.convertF32U8RoundScalar, .{ f32_u8_round, f32_src }, n);
            report("convert-f32-u8-round/scalar", n, n * 5, n * 5, result);
            result = try measure(io, kernels.convertF32U8SatScalar, .{ f32_u8_sat, f32_src }, n);
            report("convert-f32-u8-sat/scalar", n, n * 5, n * 5, result);
            result = try measure(io, kernels.f32ToU16SatScalar, .{ f32_u16_sat, f32_src }, n);
            report("convert-f32-u16-sat/scalar", n, n * 6, n * 6, result);
            result = try measure(io, kernels.narrowU16ToU8TruncScalar, .{ narrow_trunc, u16_src }, n);
            report("narrow-u16-u8-trunc/scalar", n, n * 3, n * 3, result);
            result = try measure(io, kernels.narrowU16ToU8RoundScalar, .{ narrow_round, u16_src }, n);
            report("narrow-u16-u8-round/scalar", n, n * 3, n * 3, result);
            result = try measure(io, kernels.narrowU16ToU8SatScalar, .{ narrow_sat, u16_src }, n);
            report("narrow-u16-u8-sat/scalar", n, n * 3, n * 3, result);
        }

        {
            const pack_src = try allocator.alloc(u8, n * 4);
            defer allocator.free(pack_src);
            const packed_words = try allocator.alloc(u32, n);
            defer allocator.free(packed_words);
            const unpacked = try allocator.alloc(u8, n * 4);
            defer allocator.free(unpacked);
            for (pack_src, 0..) |*value, i| {
                value.* = @truncate(i * 41 + 3);
            }
            kernels.packU8x4ToU32Scalar(packed_words, pack_src);
            kernels.unpackU32ToU8x4Scalar(unpacked, packed_words);
            for (packed_words, 0..) |value, group| {
                const offset = group * 4;
                const expected: u32 = @as(u32, pack_src[offset]) |
                    (@as(u32, pack_src[offset + 1]) << 8) |
                    (@as(u32, pack_src[offset + 2]) << 16) |
                    (@as(u32, pack_src[offset + 3]) << 24);
                std.debug.assert(value == expected);
            }
            std.debug.assert(std.mem.eql(u8, pack_src, unpacked));

            var result = try measure(io, kernels.packU8x4ToU32Scalar, .{ packed_words, pack_src }, n);
            report("pack-u8x4-u32/scalar", n, n * 8, n * 8, result);
            result = try measure(io, kernels.unpackU32ToU8x4Scalar, .{ unpacked, packed_words }, n);
            report("unpack-u32-u8x4/scalar", n, n * 8, n * 8, result);
        }

        {
            const blend_a = try allocator.alloc(u8, n);
            defer allocator.free(blend_a);
            const blend_b = try allocator.alloc(u8, n);
            defer allocator.free(blend_b);
            const blend_scalar = try allocator.alloc(u8, n);
            defer allocator.free(blend_scalar);
            const blend_vector = try allocator.alloc(u8, n);
            defer allocator.free(blend_vector);
            const convolution_src = try allocator.alloc(u8, n);
            defer allocator.free(convolution_src);
            const convolve3_scalar = try allocator.alloc(u8, n);
            defer allocator.free(convolve3_scalar);
            const convolve3_vector = try allocator.alloc(u8, n);
            defer allocator.free(convolve3_vector);
            const convolve5_scalar = try allocator.alloc(u8, n);
            defer allocator.free(convolve5_scalar);
            const convolve5_vector = try allocator.alloc(u8, n);
            defer allocator.free(convolve5_vector);

            for (blend_a, blend_b, convolution_src, 0..) |*a, *b, *source, i| {
                a.* = @truncate(i * 37 + 11);
                b.* = @truncate(i * 53 + 7);
                source.* = @truncate(i * 29 + 3);
            }
            const weight: u16 = 77;
            kernels.blendU8Scalar(blend_scalar, blend_a, blend_b, weight);
            kernels.blendU8Vector(blend_vector, blend_a, blend_b, weight);
            std.debug.assert(std.mem.eql(u8, blend_scalar, blend_vector));
            kernels.convolve3U8Scalar(convolve3_scalar, convolution_src);
            kernels.convolve3U8Vector(convolve3_vector, convolution_src);
            std.debug.assert(std.mem.eql(u8, convolve3_scalar, convolve3_vector));
            kernels.convolve5U8Scalar(convolve5_scalar, convolution_src);
            kernels.convolve5U8Vector(convolve5_vector, convolution_src);
            std.debug.assert(std.mem.eql(u8, convolve5_scalar, convolve5_vector));

            var result = try measure(io, kernels.blendU8Scalar, .{ blend_scalar, blend_a, blend_b, weight }, n);
            report("blend-u8/scalar-autovec", n, n * 3, n * 3, result);
            result = try measure(io, kernels.blendU8Vector, .{ blend_vector, blend_a, blend_b, weight }, n);
            report("blend-u8/native-vector", n, n * 3, n * 3, result);
            result = try measure(io, kernels.convolve3U8Scalar, .{ convolve3_scalar, convolution_src }, n);
            report("convolve3-u8/scalar-autovec", n, n * 2, n * 2, result);
            result = try measure(io, kernels.convolve3U8Vector, .{ convolve3_vector, convolution_src }, n);
            report("convolve3-u8/native-vector", n, n * 2, n * 2, result);
            result = try measure(io, kernels.convolve5U8Scalar, .{ convolve5_scalar, convolution_src }, n);
            report("convolve5-u8/scalar-autovec", n, n * 2, n * 2, result);
            result = try measure(io, kernels.convolve5U8Vector, .{ convolve5_vector, convolution_src }, n);
            report("convolve5-u8/native-vector", n, n * 2, n * 2, result);
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
