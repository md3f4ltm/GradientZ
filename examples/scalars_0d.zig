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

    std.debug.print("=== 0D Tensors (Scalars) Example ===\n\n", .{});

    try test0DTensors(allocator);

    std.debug.print("\n✅ 0D tensor (scalar) example completed successfully!\n", .{});
}

fn test0DTensors(allocator: std.mem.Allocator) !void {
    std.debug.print("📊 Testing 0D tensors (scalars)...\n", .{});

    // Create scalar tensors
    var scalar_a = try Tensor.init(allocator, &.{}, GradientZ.cpu());
    defer scalar_a.deinit();
    try scalar_a.setData(&.{5.0});

    var scalar_b = try Tensor.init(allocator, &.{}, GradientZ.cpu());
    defer scalar_b.deinit();
    try scalar_b.setData(&.{3.0});

    std.debug.print("   Scalar A: ", .{});
    try scalar_a.print();
    std.debug.print("   Scalar B: ", .{});
    try scalar_b.print();

    // Test scalar operations
    std.debug.print("   🔢 Testing scalar arithmetic...\n", .{});

    // Addition
    var add_result = try scalar_a.add(&scalar_b);
    defer add_result.deinit();
    std.debug.print("   A + B = ", .{});
    try add_result.print();

    // Multiplication
    var mul_result = try scalar_a.mul(&scalar_b);
    defer mul_result.deinit();
    std.debug.print("   A * B = ", .{});
    try mul_result.print();

    // Division
    var div_result = try scalar_a.div(&scalar_b);
    defer div_result.deinit();
    std.debug.print("   A / B = ", .{});
    try div_result.print();

    // Power
    var pow_result = try scalar_a.pow(&scalar_b);
    defer pow_result.deinit();
    std.debug.print("   A ^ B = ", .{});
    try pow_result.print();

    std.debug.print("   ✓ 0D tensor operations completed!\n", .{});
}
