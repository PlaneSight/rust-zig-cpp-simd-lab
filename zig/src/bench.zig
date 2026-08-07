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
    const result = @call(.auto, function, args);
    std.mem.doNotOptimizeAway(&result);
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
        }

        {
            const a = try allocator.alloc(u8, n);
            defer allocator.free(a);
            const b = try allocator.alloc(u8, n);
            defer allocator.free(b);
            for (a, b, 0..) |*av, *bv, i| {
                av.* = @intCast((i * 17 + 3) & 255);
                bv.* = @intCast((i * 29 + 11) & 255);
            }
            std.debug.assert(kernels.sadU8Scalar(a, b) == kernels.sadU8Vector(a, b));

            var result = try measure(io, kernels.sadU8Scalar, .{ a, b }, n);
            report("sad-u8/scalar-autovec", n, n * 2, n * 2, result);
            result = try measure(io, kernels.sadU8Vector, .{ a, b }, n);
            report("sad-u8/native-vector", n, n * 2, n * 2, result);
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
        "META size_count={d} warmup_samples={d} sample_count={d} target_elements_per_sample={d}\n",
        .{ sizes.len, warmup_samples, sample_count, target_elements_per_sample },
    );
}
