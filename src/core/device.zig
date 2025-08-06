const std = @import("std");

pub const DeviceType = enum {
    cpu,
    gpu,
};

pub const Device = struct {
    device_type: DeviceType,
    id: u32, // Device ID (0 for CPU, GPU index for GPU)

    const Self = @This();

    /// Create a CPU device
    pub fn cpu() Self {
        return Self{ .device_type = .cpu, .id = 0 };
    }

    /// Create a GPU device
    pub fn gpu(id: u32) Self {
        return Self{ .device_type = .gpu, .id = id };
    }

    /// Check if this is a CPU device
    pub fn isCpu(self: Self) bool {
        return self.device_type == .cpu;
    }

    /// Check if this is a GPU device
    pub fn isGpu(self: Self) bool {
        return self.device_type == .gpu;
    }

    /// Check if two devices are equal
    pub fn eql(self: Self, other: Self) bool {
        return self.device_type == other.device_type and self.id == other.id;
    }

    /// Format device for printing
    pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self.device_type) {
            .cpu => try writer.print("CPU", .{}),
            .gpu => try writer.print("GPU:{}", .{self.id}),
        }
    }
};

// Global default device
var default_device: Device = Device.cpu();

// Get the default device
pub fn getDefaultDevice() Device {
    return default_device;
}

// Set the default device
pub fn setDefaultDevice(device: Device) void {
    default_device = device;
}

test "device creation and properties" {
    const cpu_dev = Device.cpu();
    const gpu_dev = Device.gpu(0);

    try std.testing.expect(cpu_dev.isCpu());
    try std.testing.expect(!cpu_dev.isGpu());
    try std.testing.expect(!gpu_dev.isCpu());
    try std.testing.expect(gpu_dev.isGpu());

    try std.testing.expect(cpu_dev.eql(Device.cpu()));
    try std.testing.expect(!cpu_dev.eql(gpu_dev));
}
