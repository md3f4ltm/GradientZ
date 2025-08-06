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

    std.debug.print("=== 2D Tensors (Matrices) Example ===\n\n", .{});

    try test2DTensors(allocator);

    std.debug.print("\n✅ 2D tensor (matrix) example completed successfully!\n", .{});
}

fn test2DTensors(allocator: std.mem.Allocator) !void {
    std.debug.print("📋 Testing 2D tensors (matrices)...\n", .{});
    const device = GradientZ.cpu();

    var matrix_a = try Tensor.init(allocator, &.{ 2, 3 }, device);
    defer matrix_a.deinit();
    try matrix_a.setData(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    var matrix_b = try Tensor.init(allocator, &.{ 2, 3 }, device);
    defer matrix_b.deinit();
    try matrix_b.setData(&.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });

    std.debug.print("   Matrix A (2x3):\n", .{});
    try matrix_a.print();
    std.debug.print("   Matrix B (2x3):\n", .{});
    try matrix_b.print();

    // Test matrix operations (element-wise with same shapes)
    std.debug.print("   🔢 Testing matrix arithmetic...\n", .{});

    // Addition
    var add_result = try matrix_a.add(&matrix_b);
    defer add_result.deinit();
    std.debug.print("   A + B =\n", .{});
    try add_result.print();

    // Element-wise multiplication
    var mul_result = try matrix_a.mul(&matrix_b);
    defer mul_result.deinit();
    std.debug.print("   A * B (element-wise) =\n", .{});
    try mul_result.print();

    // Division
    var div_result = try matrix_a.div(&matrix_b);
    defer div_result.deinit();
    std.debug.print("   A / B =\n", .{});
    try div_result.print();

    // Power
    var pow_result = try matrix_a.pow(&matrix_b);
    defer pow_result.deinit();
    std.debug.print("   A ^ B =\n", .{});
    try pow_result.print();

    // Test matrix multiplication (create compatible matrix for matmul)
    var matmul_c = try Tensor.init(allocator, &.{ 3, 2 }, device);
    defer matmul_c.deinit();
    try matmul_c.setData(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    var matmul_result = try matrix_a.matmul(&matmul_c);
    defer matmul_result.deinit();
    std.debug.print("   A @ C (2x3 @ 3x2 = 2x2) =\n", .{});
    try matmul_result.print();

    // Test convenience functions
    var zeros_2d = try GradientZ.zeros(allocator, &.{ 2, 2 }, GradientZ.cpu());
    defer zeros_2d.deinit();
    std.debug.print("   Zeros matrix (2x2):\n", .{});
    try zeros_2d.print();

    var ones_2d = try GradientZ.ones(allocator, &.{ 2, 2 }, GradientZ.cpu());
    defer ones_2d.deinit();
    std.debug.print("   Ones matrix (2x2):\n", .{});
    try ones_2d.print();

    std.debug.print("   ✓ 2D tensor operations completed!\n", .{});
}
