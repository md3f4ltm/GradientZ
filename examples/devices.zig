const std = @import("std");
const GradientZ = @import("GradientZ_lib");
const Tensor = GradientZ.Tensor;

pub fn main() !void {
    // Initialize the library
    GradientZ.init();

    // Create allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Device Functionality Example ===\n\n", .{});

    try testDevices(allocator);

    std.debug.print("\n✅ Device example completed successfully!\n", .{});
}

fn testDevices(allocator: std.mem.Allocator) !void {
    std.debug.print("🖥️  Testing device functionality...\n", .{});

    // Test CPU device with different tensor dimensions
    const cpu_device = GradientZ.cpu();

    std.debug.print("📱 CPU Device Tests:\n", .{});
    var cpu_scalar = try Tensor.init(allocator, &.{}, cpu_device);
    defer cpu_scalar.deinit();
    try cpu_scalar.setData(&.{42.0});
    std.debug.print("   CPU scalar: ", .{});
    try cpu_scalar.print();

    var cpu_vector = try GradientZ.ones(allocator, &.{3}, cpu_device);
    defer cpu_vector.deinit();
    std.debug.print("   CPU vector: ", .{});
    try cpu_vector.print();

    var cpu_matrix = try GradientZ.ones(allocator, &.{ 2, 2 }, cpu_device);
    defer cpu_matrix.deinit();
    std.debug.print("   CPU matrix:\n", .{});
    try cpu_matrix.print();

    var cpu_3d = try GradientZ.ones(allocator, &.{ 2, 2, 2 }, cpu_device);
    defer cpu_3d.deinit();
    std.debug.print("   CPU 3D tensor:\n", .{});
    try cpu_3d.print();

    std.debug.print("   ✅ CPU device works with all tensor dimensions\n", .{});

    // Note: GPU backend not implemented yet
    std.debug.print("   ℹ️  GPU backend: Not implemented yet\n", .{});
    std.debug.print("   ℹ️  ROCm backend: Not implemented yet\n", .{});

    // Test device comparison
    const cpu1 = GradientZ.cpu();
    const cpu2 = GradientZ.cpu();

    if (cpu1.eql(cpu2)) {
        std.debug.print("   ✓ Device equality works\n", .{});
    }

    std.debug.print("   ✓ CPU backend fully functional with multi-dimensional tensors\n", .{});
}
