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

    std.debug.print("=== 3D Tensors Example ===\n\n", .{});

    try test3DTensors(allocator);

    std.debug.print("\n✅ 3D tensor example completed successfully!\n", .{});
}

fn test3DTensors(allocator: std.mem.Allocator) !void {
    std.debug.print("📦 Testing 3D tensors...\n", .{});
    const device = GradientZ.cpu();

    var tensor_3d_a = try Tensor.init(allocator, &.{ 2, 2, 3 }, device);
    defer tensor_3d_a.deinit();
    try tensor_3d_a.setData(&.{
        1.0, 2.0, 3.0, // [0][0]
        4.0, 5.0, 6.0, // [0][1]
        7.0, 8.0, 9.0, // [1][0]
        10.0, 11.0, 12.0, // [1][1]
    });

    var tensor_3d_b = try Tensor.init(allocator, &.{ 2, 2, 3 }, device);
    defer tensor_3d_b.deinit();
    try tensor_3d_b.setData(&.{
        13.0, 14.0, 15.0, // [0][0]
        16.0, 17.0, 18.0, // [0][1]
        19.0, 20.0, 21.0, // [1][0]
        22.0, 23.0, 24.0, // [1][1]
    });

    std.debug.print("   3D Tensor A (2x2x3):\n", .{});
    try tensor_3d_a.print();
    std.debug.print("   3D Tensor B (2x2x3):\n", .{});
    try tensor_3d_b.print();

    // Test 3D tensor operations
    std.debug.print("   🔢 Testing 3D tensor arithmetic...\n", .{});

    // Addition
    var add_result = try tensor_3d_a.add(&tensor_3d_b);
    defer add_result.deinit();
    std.debug.print("   A + B =\n", .{});
    try add_result.print();

    // Element-wise multiplication
    var mul_result = try tensor_3d_a.mul(&tensor_3d_b);
    defer mul_result.deinit();
    std.debug.print("   A * B (element-wise) =\n", .{});
    try mul_result.print();

    // Division
    var div_result = try tensor_3d_a.div(&tensor_3d_b);
    defer div_result.deinit();
    std.debug.print("   A / B =\n", .{});
    try div_result.print();

    // Power
    var pow_result = try tensor_3d_a.pow(&tensor_3d_b);
    defer pow_result.deinit();
    std.debug.print("   A ^ B =\n", .{});
    try pow_result.print();

    // Test convenience functions
    var zeros_3d = try GradientZ.zeros(allocator, &.{ 2, 2, 2 }, GradientZ.cpu());
    defer zeros_3d.deinit();
    std.debug.print("   Zeros 3D tensor (2x2x2):\n", .{});
    try zeros_3d.print();

    var ones_3d = try GradientZ.ones(allocator, &.{ 2, 2, 2 }, GradientZ.cpu());
    defer ones_3d.deinit();
    std.debug.print("   Ones 3D tensor (2x2x2):\n", .{});
    try ones_3d.print();

    std.debug.print("   ✓ 3D tensor operations completed!\n", .{});
}
