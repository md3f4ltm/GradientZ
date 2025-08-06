const std = @import("std");
const GradientZ = @import("GradientZ_lib");

pub fn main() !void {
    // Initialize the library
    GradientZ.init();

    // Create allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== GradientZ Tensor Library ===\n\n", .{});

    // Show library information
    GradientZ.info();

    std.debug.print("\n📚 Examples have been moved to the examples/ folder!\n", .{});
    std.debug.print("Run them with:\n", .{});
    std.debug.print("  zig build example-scalars      # 0D tensors (scalars)\n", .{});
    std.debug.print("  zig build example-vectors      # 1D tensors (vectors)\n", .{});
    std.debug.print("  zig build example-matrices     # 2D tensors (matrices)\n", .{});
    std.debug.print("  zig build example-3d           # 3D tensors\n", .{});
    std.debug.print("  zig build example-autograd     # Automatic differentiation\n", .{});
    std.debug.print("  zig build example-mlp          # Multi-Layer Perceptron\n", .{});
    std.debug.print("  zig build example-devices      # Device functionality\n", .{});
    std.debug.print("  zig build example-training     # Training with gradient descent\n", .{});
    std.debug.print("  zig build example              # PyTorch-like API example\n", .{});
    std.debug.print("  zig build examples             # Run all examples\n", .{});

    std.debug.print("\n🚀 Quick demo - Creating tensors:\n", .{});

    // Quick demo of basic functionality
    var zeros_tensor = try GradientZ.zeros(allocator, &.{ 2, 3 }, GradientZ.cpu());
    defer zeros_tensor.deinit();

    std.debug.print("Zeros tensor (2x3):\n", .{});
    try zeros_tensor.print();

    var ones_tensor = try GradientZ.ones(allocator, &.{ 2, 3 }, GradientZ.cpu());
    defer ones_tensor.deinit();

    std.debug.print("\nOnes tensor (2x3):\n", .{});
    try ones_tensor.print();

    // Basic arithmetic
    var sum = try zeros_tensor.add(&ones_tensor);
    defer sum.deinit();

    std.debug.print("\nZeros + Ones:\n", .{});
    try sum.print();

    std.debug.print("\n✅ GradientZ library is working! Explore the examples to learn more.\n", .{});
}
