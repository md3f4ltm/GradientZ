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

    std.debug.print("=== Multi-Layer Perceptron (MLP) Example ===\n\n", .{});

    try testMLP(allocator);

    std.debug.print("\n✅ MLP example completed successfully!\n", .{});
}

fn testMLP(allocator: std.mem.Allocator) !void {
    std.debug.print("🧠 Testing Multi-Layer Perceptron (MLP)...\n", .{});

    const cpu_device = GradientZ.cpu();

    // Create a simple MLP with 2 layers: input(2) -> hidden(3) -> output(1)
    std.debug.print("   Creating MLP weights and biases...\n", .{});

    // Layer 1: input(2) -> hidden(3)
    var w1 = try Tensor.initWithGrad(allocator, &.{ 2, 3 }, cpu_device, true);
    defer w1.deinit();
    try w1.setData(&.{ 0.5, -0.3, 0.8, 0.2, -0.6, 0.4 }); // 2x3 matrix

    var b1 = try Tensor.initWithGrad(allocator, &.{ 1, 3 }, cpu_device, true);
    defer b1.deinit();
    try b1.setData(&.{ 0.1, -0.2, 0.3 }); // 1x3 bias

    // Layer 2: hidden(3) -> output(1)
    var w2 = try Tensor.initWithGrad(allocator, &.{ 3, 1 }, cpu_device, true);
    defer w2.deinit();
    try w2.setData(&.{ 0.7, -0.4, 0.9 }); // 3x1 matrix

    var b2 = try Tensor.initWithGrad(allocator, &.{ 1, 1 }, cpu_device, true);
    defer b2.deinit();
    try b2.setData(&.{0.2}); // 1x1 bias

    // Create input data: batch_size=1, input_features=2
    std.debug.print("   Creating input data...\n", .{});
    var input = try Tensor.init(allocator, &.{ 1, 2 }, cpu_device);
    defer input.deinit();
    try input.setData(&.{ 1.0, -0.5 }); // Sample input

    std.debug.print("   Input: ", .{});
    try input.print();

    // Forward pass
    std.debug.print("   Forward pass:\n", .{});

    // Layer 1: z1 = input @ w1 + b1
    var z1 = try input.matmul(&w1);
    defer z1.deinit();
    var z1_with_bias = try z1.add(&b1);
    defer z1_with_bias.deinit();

    std.debug.print("     Layer 1 pre-activation: ", .{});
    try z1_with_bias.print();

    // Apply ReLU activation
    var h1 = try z1_with_bias.relu();
    defer h1.deinit();

    std.debug.print("     Layer 1 post-ReLU: ", .{});
    try h1.print();

    // Layer 2: z2 = h1 @ w2 + b2
    var z2 = try h1.matmul(&w2);
    defer z2.deinit();
    var output = try z2.add(&b2);
    defer output.deinit();

    std.debug.print("     Final output: ", .{});
    try output.print();

    // Apply sigmoid to output for binary classification
    var sigmoid_output = try output.sigmoid();
    defer sigmoid_output.deinit();

    std.debug.print("     Sigmoid output: ", .{});
    try sigmoid_output.print();

    // Simulate a target for loss computation
    var target = try Tensor.init(allocator, &.{ 1, 1 }, cpu_device);
    defer target.deinit();
    try target.setData(&.{1.0}); // Target value

    std.debug.print("     Target: ", .{});
    try target.print();

    // Compute simple squared loss: (output - target)^2
    var diff = try sigmoid_output.sub(&target);
    defer diff.deinit();
    var loss = try diff.mul(&diff);
    defer loss.deinit();

    std.debug.print("     Loss (squared error): ", .{});
    try loss.print();

    // Test backward pass if gradients are enabled
    if (w1.requires_grad and w2.requires_grad) {
        std.debug.print("   Testing backward pass...\n", .{});
        try loss.backward();

        if (w1.grad) |grad1| {
            std.debug.print("     W1 gradients: ", .{});
            try grad1.print();
        }

        if (w2.grad) |grad2| {
            std.debug.print("     W2 gradients: ", .{});
            try grad2.print();
        }

        if (b1.grad) |grad_b1| {
            std.debug.print("     B1 gradients: ", .{});
            try grad_b1.print();
        }

        if (b2.grad) |grad_b2| {
            std.debug.print("     B2 gradients: ", .{});
            try grad_b2.print();
        }
    }

    std.debug.print("   ✅ MLP test completed successfully!\n", .{});
}
