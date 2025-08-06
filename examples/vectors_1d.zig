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

    std.debug.print("=== 1D Tensors (Vectors) Example ===\n\n", .{});

    try test1DTensors(allocator);

    std.debug.print("\n✅ 1D tensor (vector) example completed successfully!\n", .{});
}

fn test1DTensors(allocator: std.mem.Allocator) !void {
    std.debug.print("📈 Testing 1D tensors (vectors)...\n", .{});

    // Create 1D tensors
    // Create vector tensors
    var vector_a = try Tensor.init(allocator, &.{4}, GradientZ.cpu());
    defer vector_a.deinit();
    try vector_a.setData(&.{ 1.0, 2.0, 3.0, 4.0 });

    var vector_b = try Tensor.init(allocator, &.{4}, GradientZ.cpu());
    defer vector_b.deinit();
    try vector_b.setData(&.{ 5.0, 6.0, 7.0, 8.0 });

    std.debug.print("   Vector A: ", .{});
    try vector_a.print();
    std.debug.print("   Vector B: ", .{});
    try vector_b.print();

    // Test vector operations
    std.debug.print("   🔢 Testing vector arithmetic...\n", .{});

    // Addition
    var add_result = try vector_a.add(&vector_b);
    defer add_result.deinit();
    std.debug.print("   A + B = ", .{});
    try add_result.print();

    // Element-wise multiplication
    var mul_result = try vector_a.mul(&vector_b);
    defer mul_result.deinit();
    std.debug.print("   A * B (element-wise) = ", .{});
    try mul_result.print();

    // Division
    var div_result = try vector_a.div(&vector_b);
    defer div_result.deinit();
    std.debug.print("   A / B = ", .{});
    try div_result.print();

    // Power
    var pow_result = try vector_a.pow(&vector_b);
    defer pow_result.deinit();
    std.debug.print("   A ^ B = ", .{});
    try pow_result.print();

    // Test convenience functions
    var zeros_1d = try GradientZ.zeros(allocator, &.{3}, GradientZ.cpu());
    defer zeros_1d.deinit();
    std.debug.print("   Zeros vector: ", .{});
    try zeros_1d.print();

    var ones_1d = try GradientZ.ones(allocator, &.{3}, GradientZ.cpu());
    defer ones_1d.deinit();
    std.debug.print("   Ones vector: ", .{});
    try ones_1d.print();

    std.debug.print("   ✓ 1D tensor operations completed!\n", .{});
}
