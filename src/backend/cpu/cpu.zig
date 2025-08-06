const std = @import("std");
const Backend = @import("../backend.zig").Backend;
const BackendError = @import("../backend.zig").BackendError;
const BackendMemory = @import("../backend.zig").BackendMemory;
const Device = @import("../../core/device.zig").Device;

/// CPU backend implementation
const CpuBackend = struct {
    /// Allocate memory on CPU
    fn alloc(allocator: std.mem.Allocator, device: Device, bytes: usize) BackendError!BackendMemory {
        if (!device.isCpu()) return BackendError.UnsupportedDevice;

        const ptr = allocator.alloc(u8, bytes) catch |err| switch (err) {
            error.OutOfMemory => return BackendError.OutOfMemory,
        };

        return BackendMemory{
            .ptr = ptr,
            .device = device,
        };
    }

    /// Free CPU memory
    fn free(allocator: std.mem.Allocator, memory: BackendMemory) void {
        allocator.free(memory.ptr);
    }

    /// Copy memory on CPU
    fn copy(dst: BackendMemory, src: BackendMemory, bytes: usize) BackendError!void {
        if (dst.ptr.len < bytes or src.ptr.len < bytes) {
            return BackendError.CopyFailed;
        }
        @memcpy(dst.ptr[0..bytes], src.ptr[0..bytes]);
    }

    /// CPU element-wise addition for f32
    fn add(result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        if (result_data.len < num_elements or a_data.len < num_elements or b_data.len < num_elements) {
            return BackendError.ComputeFailed;
        }

        for (0..num_elements) |i| {
            result_data[i] = a_data[i] + b_data[i];
        }
    }
    /// CPU matrix multiplication
    fn matmul(result: BackendMemory, a: BackendMemory, b: BackendMemory, m: usize, k: usize, n: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        if (result_data.len < m * n or a_data.len < m * k or b_data.len < k * n) {
            return BackendError.ComputeFailed;
        }

        // Init result data to zero
        for (result_data[0 .. m * n]) |*val| {
            val.* = 0.0;
        }
        // Perform matrix multiplication
        for (0..m) |i| {
            for (0..n) |j| {
                for (0..k) |ki| {
                    result_data[i * n + j] += a_data[i * k + ki] * b_data[ki * n + j];
                }
            }
        }
    }

    fn div(result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        for (0..num_elements) |i| {
            if (b_data[i] == 0.0) {
                return BackendError.ComputeFailed; // Handle div by zero
            }
            result_data[i] = a_data[i] / b_data[i];
        }
    }
    fn mul(result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = a_data[i] * b_data[i];
        }
    }
    /// CPU broadcasted addition for f32
    fn addBroadcast(result: BackendMemory, a: BackendMemory, b: BackendMemory, result_shape: []const usize, a_shape: []const usize, b_shape: []const usize) BackendError!void {
        _ = a_shape; // Mark as unused
        _ = b_shape; // Mark as unused

        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        var result_elements: usize = 1;
        for (result_shape) |dim| result_elements *= dim;

        if (result_data.len < result_elements) return BackendError.ComputeFailed;

        for (0..result_elements) |i| {
            result_data[i] = a_data[0] + b_data[0];
        }
    }
    fn pow(result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = std.math.pow(f32, a_data[i], b_data[i]);
        }
    }

    /// CPU element-wise subtraction
    fn sub(result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);
        const b_data = b.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = a_data[i] - b_data[i];
        }
    }

    /// CPU element-wise negation
    fn neg(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = -a_data[i];
        }
    }

    /// CPU ReLU activation
    fn relu(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = @max(0.0, a_data[i]);
        }
    }

    /// CPU tanh activation
    fn tanh_impl(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = std.math.tanh(a_data[i]);
        }
    }

    /// CPU sigmoid activation
    fn sigmoid(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = 1.0 / (1.0 + std.math.exp(-a_data[i]));
        }
    }

    /// CPU exponential
    fn exp(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = std.math.exp(a_data[i]);
        }
    }

    /// CPU natural logarithm
    fn log(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            if (a_data[i] <= 0.0) {
                return BackendError.ComputeFailed; // Handle invalid input
            }
            result_data[i] = std.math.log(f32, std.math.e, a_data[i]);
        }
    }

    /// CPU sine
    fn sin(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = std.math.sin(a_data[i]);
        }
    }

    /// CPU cosine
    fn cos(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = std.math.cos(a_data[i]);
        }
    }

    /// CPU square root
    fn sqrt(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            if (a_data[i] < 0.0) {
                return BackendError.ComputeFailed; // Handle negative input
            }
            result_data[i] = std.math.sqrt(a_data[i]);
        }
    }

    /// CPU absolute value
    fn abs(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..num_elements) |i| {
            result_data[i] = @abs(a_data[i]);
        }
    }

    /// CPU sum reduction
    fn sum(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!f32 {
        _ = result; // Not used for reductions
        const a_data = a.asType(f32);

        var total: f32 = 0.0;
        for (0..num_elements) |i| {
            total += a_data[i];
        }
        return total;
    }

    /// CPU mean reduction
    fn mean(result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!f32 {
        _ = result; // Not used for reductions
        const a_data = a.asType(f32);

        var total: f32 = 0.0;
        for (0..num_elements) |i| {
            total += a_data[i];
        }
        return total / @as(f32, @floatFromInt(num_elements));
    }

    /// CPU matrix transpose
    fn transpose(result: BackendMemory, a: BackendMemory, rows: usize, cols: usize) BackendError!void {
        const result_data = result.asType(f32);
        const a_data = a.asType(f32);

        for (0..rows) |i| {
            for (0..cols) |j| {
                result_data[j * rows + i] = a_data[i * cols + j];
            }
        }
    }
};

/// Get CPU backend instance
pub fn getCpuBackend() Backend {
    return Backend{
        .allocFn = CpuBackend.alloc,
        .freeFn = CpuBackend.free,
        .copyFn = CpuBackend.copy,
        .addFn = CpuBackend.add,
        .matmulFn = CpuBackend.matmul,
        .divFn = CpuBackend.div,
        .mulFn = CpuBackend.mul,
        .powFn = CpuBackend.pow,
        .subFn = CpuBackend.sub,
        .negFn = CpuBackend.neg,
        .reluFn = CpuBackend.relu,
        .tanhFn = CpuBackend.tanh_impl,
        .sigmoidFn = CpuBackend.sigmoid,
        .expFn = CpuBackend.exp,
        .logFn = CpuBackend.log,
        .sinFn = CpuBackend.sin,
        .cosFn = CpuBackend.cos,
        .sqrtFn = CpuBackend.sqrt,
        .absFn = CpuBackend.abs,
        .sumFn = CpuBackend.sum,
        .meanFn = CpuBackend.mean,
        .transposeFn = CpuBackend.transpose,
        .addBroadcastFn = CpuBackend.addBroadcast,
    };
}

test "cpu backend operations" {
    const backend = getCpuBackend();
    const allocator = std.testing.allocator;
    const device = Device.cpu();

    // Test allocation
    var mem1 = try backend.alloc(allocator, device, 4 * @sizeOf(f32));
    defer backend.free(allocator, mem1);

    var mem2 = try backend.alloc(allocator, device, 4 * @sizeOf(f32));
    defer backend.free(allocator, mem2);

    var result = try backend.alloc(allocator, device, 4 * @sizeOf(f32));
    defer backend.free(allocator, result);

    // Set up test data
    const data1 = mem1.asType(f32);
    const data2 = mem2.asType(f32);
    data1[0] = 1.0;
    data1[1] = 2.0;
    data1[2] = 3.0;
    data1[3] = 4.0;
    data2[0] = 5.0;
    data2[1] = 6.0;
    data2[2] = 7.0;
    data2[3] = 8.0;

    // Test addition
    try backend.add(result, mem1, mem2, 4);

    const result_data = result.asType(f32);
    try std.testing.expectEqual(@as(f32, 6.0), result_data[0]);
    try std.testing.expectEqual(@as(f32, 8.0), result_data[1]);
    try std.testing.expectEqual(@as(f32, 10.0), result_data[2]);
    try std.testing.expectEqual(@as(f32, 12.0), result_data[3]);
}
