//! GradientZ - A tensor computation library for Zig
//! Supports CPU and GPU (ROCm/HIP) backends

const std = @import("std");

// Core modules
const tensor_mod = @import("core/tensor.zig");
const device_mod = @import("core/device.zig");
const shape_mod = @import("core/shape.zig");
const autograd_mod = @import("core/autograd.zig");

// Backend modules
const backend_mod = @import("backend/backend.zig");
const cpu_backend = @import("backend/cpu/cpu.zig");

// Neural network module
const nn_mod = @import("nn/nn.zig");

// Re-export core types and functions
pub const Tensor = tensor_mod.Tensor;
pub const Device = device_mod.Device;
pub const zeros = tensor_mod.zeros;
pub const ones = tensor_mod.ones;
pub const zerosWithGrad = tensor_mod.zerosWithGrad;
pub const onesWithGrad = tensor_mod.onesWithGrad;

// Re-export errors
pub const TensorError = tensor_mod.TensorError;
pub const BackendError = backend_mod.BackendError;
pub const ShapeError = shape_mod.ShapeError;

// Re-export autograd types
pub const GradFn = autograd_mod.GradFn;

// Re-export neural network types
pub const nn = nn_mod;
// Library state
var initialized = false;
var rocm_available = false;

/// Initialize the GradientZ library
/// This registers available backends and sets up the library
pub fn init() void {
    if (initialized) return;

    // Always register CPU backend
    const cpu_be = cpu_backend.getCpuBackend();
    backend_mod.registerCpuBackend(cpu_be);

    // Try to register ROCm backend if available
    // For now, we'll mark it as unavailable since we don't have ROCm implementation
    rocm_available = false; // TODO: Detect and register ROCm backend

    initialized = true;
}

/// Register CPU backend (convenience function)
pub fn registerCpuBackend() void {
    const cpu_be = cpu_backend.getCpuBackend();
    backend_mod.registerCpuBackend(cpu_be);
}

/// Print library information
pub fn info() void {
    const stdout = std.io.getStdOut().writer();
    stdout.print("🚀 GradientZ Tensor Library\n", .{}) catch {};
    stdout.print("   Version: 0.1.0\n", .{}) catch {};
    stdout.print("   Backends:\n", .{}) catch {};
    stdout.print("     ✅ CPU (sequential) - WORKING\n", .{}) catch {};
    stdout.print("     🚧 ROCm/HIP (AMD GPU) - NOT IMPLEMENTED\n", .{}) catch {};
    stdout.print("     🚧 CUDA (NVIDIA GPU) - NOT IMPLEMENTED\n", .{}) catch {};
}

/// Check if ROCm backend is available
/// Currently always returns false as ROCm backend is not implemented yet
pub fn isRocmAvailable() bool {
    return false; // TODO: Implement ROCm backend
}

/// Get library version
pub fn version() []const u8 {
    return "0.1.0";
}

/// Device creation helpers with updated API
pub const DeviceType = device_mod.DeviceType;

// Update the Device API to match what main.zig expects
pub const DeviceAPI = struct {
    /// Create a CPU device
    pub fn cpu() Device {
        return Device.cpu();
    }

    /// Create a ROCm device (AMD GPU) - NOT IMPLEMENTED YET
    /// This creates a device struct but the backend won't work
    pub fn rocm(id: u32) Device {
        return Device{ .device_type = .gpu, .id = id }; // Will fail at runtime - no backend
    }

    /// Create a GPU device (generic) - NOT IMPLEMENTED YET
    /// This creates a device struct but the backend won't work
    pub fn gpu(id: u32) Device {
        return Device.gpu(id); // Will fail at runtime - no backend
    }
};

// Export device creation functions at top level
pub const device = DeviceAPI;

// Also export individual functions for convenience
pub fn cpu() Device {
    return Device.cpu();
}

pub fn gpu(id: u32) Device {
    return Device.gpu(id);
}

/// Create a ROCm device - NOT IMPLEMENTED YET
/// This is just a placeholder, actual ROCm operations will fail
pub fn rocm(id: u32) Device {
    return Device{ .device_type = .gpu, .id = id };
}

test "library initialization" {
    init();
    try std.testing.expect(initialized);
}

test "tensor creation through library interface" {
    init();
    const allocator = std.testing.allocator;

    // Test zeros function
    var tensor = try zeros(allocator, &.{ 2, 2 }, cpu());
    defer tensor.deinit();

    const data = try tensor.data();
    for (data) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}
