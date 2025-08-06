const std = @import("std");
const Device = @import("../core/device.zig").Device;

pub const BackendError = error{
    UnsupportedDevice,
    AllocationFailed,
    CopyFailed,
    ComputeFailed,
    OutOfMemory,
};

/// Memoru handle for backend allocated memory
pub const BackendMemory = struct {
    ptr: []u8,
    device: Device,

    const Self = @This();

    pub fn asType(self: Self, comptime T: type) []T {
        const byte_len = self.ptr.len;
        const elem_count = byte_len / @sizeOf(T);
        return @alignCast(std.mem.bytesAsSlice(T, self.ptr[0 .. elem_count * @sizeOf(T)]));
    }
};

/// Backend interface - all backends must implement this
pub const Backend = struct {
    allocFn: *const fn (std.mem.Allocator, Device, usize) BackendError!BackendMemory,
    freeFn: *const fn (std.mem.Allocator, BackendMemory) void,
    copyFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    addFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize) BackendError!void,
    matmulFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize, usize, usize) BackendError!void,
    addBroadcastFn: *const fn (BackendMemory, BackendMemory, BackendMemory, []const usize, []const usize, []const usize) BackendError!void,
    divFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize) BackendError!void,
    mulFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize) BackendError!void,
    powFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize) BackendError!void,
    subFn: *const fn (BackendMemory, BackendMemory, BackendMemory, usize) BackendError!void,
    negFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    reluFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    tanhFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    sigmoidFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    expFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    logFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    sinFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    cosFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    sqrtFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    absFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!void,
    sumFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!f32,
    meanFn: *const fn (BackendMemory, BackendMemory, usize) BackendError!f32,
    transposeFn: *const fn (BackendMemory, BackendMemory, usize, usize) BackendError!void,

    const Self = @This();

    /// Allocate memory on device
    pub fn alloc(self: Self, allocator: std.mem.Allocator, device: Device, bytes: usize) BackendError!BackendMemory {
        return self.allocFn(allocator, device, bytes);
    }

    /// Free memory
    pub fn free(self: Self, allocator: std.mem.Allocator, memory: BackendMemory) void {
        self.freeFn(allocator, memory);
    }

    /// Copy data between memory locations
    pub fn copy(self: Self, dst: BackendMemory, src: BackendMemory, bytes: usize) BackendError!void {
        return self.copyFn(dst, src, bytes);
    }

    /// Element-wise addition: result = a + b
    pub fn add(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        return self.addFn(result, a, b, num_elements);
    }

    /// Matrix multiplication: result = a * b
    pub fn matmul(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, m: usize, n: usize, k: usize) BackendError!void {
        return self.matmulFn(result, a, b, m, n, k);
    }

    /// Broadcasted addition: result = a + b with broadcasting
    pub fn addBroadcast(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, result_shape: []const usize, a_shape: []const usize, b_shape: []const usize) BackendError!void {
        return self.addBroadcastFn(result, a, b, result_shape, a_shape, b_shape);
    }
    /// Element-wise division: result = a / b
    pub fn div(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        return self.divFn(result, a, b, num_elements);
    }
    /// Element-wise multiplication: result = a * b
    pub fn mul(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        return self.mulFn(result, a, b, num_elements);
    }
    /// Element-wise power: result = a ^ b
    pub fn pow(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        return self.powFn(result, a, b, num_elements);
    }

    /// Element-wise subtraction: result = a - b
    pub fn sub(self: Self, result: BackendMemory, a: BackendMemory, b: BackendMemory, num_elements: usize) BackendError!void {
        return self.subFn(result, a, b, num_elements);
    }

    /// Element-wise negation: result = -a
    pub fn neg(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.negFn(result, a, num_elements);
    }

    /// Element-wise ReLU: result = max(0, a)
    pub fn relu(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.reluFn(result, a, num_elements);
    }

    /// Element-wise tanh: result = tanh(a)
    pub fn tanh(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.tanhFn(result, a, num_elements);
    }

    /// Element-wise sigmoid: result = 1 / (1 + exp(-a))
    pub fn sigmoid(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.sigmoidFn(result, a, num_elements);
    }

    /// Element-wise exponential: result = exp(a)
    pub fn exp(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.expFn(result, a, num_elements);
    }

    /// Element-wise natural logarithm: result = log(a)
    pub fn log(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.logFn(result, a, num_elements);
    }

    /// Element-wise sine: result = sin(a)
    pub fn sin(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.sinFn(result, a, num_elements);
    }

    /// Element-wise cosine: result = cos(a)
    pub fn cos(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.cosFn(result, a, num_elements);
    }

    /// Element-wise square root: result = sqrt(a)
    pub fn sqrt(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.sqrtFn(result, a, num_elements);
    }

    /// Element-wise absolute value: result = abs(a)
    pub fn abs(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!void {
        return self.absFn(result, a, num_elements);
    }

    /// Sum all elements: returns sum of all elements in tensor
    pub fn sum(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!f32 {
        return self.sumFn(result, a, num_elements);
    }

    /// Mean of all elements: returns mean of all elements in tensor
    pub fn mean(self: Self, result: BackendMemory, a: BackendMemory, num_elements: usize) BackendError!f32 {
        return self.meanFn(result, a, num_elements);
    }

    /// Matrix transpose: result = transpose(a) for 2D matrices
    pub fn transpose(self: Self, result: BackendMemory, a: BackendMemory, rows: usize, cols: usize) BackendError!void {
        return self.transposeFn(result, a, rows, cols);
    }
};

/// Global backend registry
var global_cpu_backend: ?Backend = null;
var global_rocm_backend: ?Backend = null;

/// Register CPU backend
pub fn registerCpuBackend(backend: Backend) void {
    global_cpu_backend = backend;
}

/// Register ROCm/HIP backend for AMD GPUs
pub fn registerRocmBackend(backend: Backend) void {
    global_rocm_backend = backend;
}

/// Get backend for device
pub fn getBackend(device: Device) BackendError!Backend {
    return switch (device.device_type) {
        .cpu => global_cpu_backend orelse return BackendError.UnsupportedDevice,
        .gpu => global_rocm_backend orelse return BackendError.UnsupportedDevice,
    };
}
