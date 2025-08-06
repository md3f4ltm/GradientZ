//! Example demonstrating PyTorch-like neural network API in GradientZ
//! This shows how to create and use neural networks similar to PyTorch

const std = @import("std");
const lib = @import("GradientZ_lib");
const Tensor = lib.Tensor;
const Device = lib.Device;
const nn = lib.nn;

/// Example equivalent to this PyTorch code:
/// ```python
/// import torch
/// import torch.nn as nn
///
/// class SimpleMLP(nn.Module):
///     def __init__(self, input_size, hidden_size, output_size):
///         super(SimpleMLP, self).__init__()
///         self.fc1 = nn.Linear(input_size, hidden_size)
///         self.fc2 = nn.Linear(hidden_size, output_size)
///         self.relu = nn.ReLU()
///
///     def forward(self, x):
///         x = self.fc1(x)
///         x = self.relu(x)
///         x = self.fc2(x)
///         return x
///
/// # Usage
/// model = SimpleMLP(2, 3, 1)
/// input_data = torch.tensor([0.5, -0.3])
/// output = model(input_data)
/// ```
pub fn main() !void {
    // Initialize the library
    lib.init();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const device = lib.cpu();

    std.debug.print("🚀 PyTorch-like Neural Network Example\n", .{});
    std.debug.print("=====================================\n\n", .{});

    // Example 1: Simple Linear Layer
    std.debug.print("📋 Example 1: Linear Layer\n", .{});
    std.debug.print("---------------------------\n", .{});

    var linear = try nn.Linear.init(allocator, 3, 2, device, true);
    defer linear.deinit();

    var input = try Tensor.init(allocator, &.{3}, device);
    defer input.deinit();
    try input.setData(&.{ 1.0, 2.0, 3.0 });

    std.debug.print("Input: ", .{});
    try input.print();

    var output = try linear.forward(&input);
    defer output.deinit();

    std.debug.print("Linear Layer Output: ", .{});
    try output.print();
    std.debug.print("\n", .{});

    // Example 2: ReLU Activation
    std.debug.print("⚡ Example 2: ReLU Activation\n", .{});
    std.debug.print("-----------------------------\n", .{});

    var relu = nn.ReLU.init(allocator, device);
    defer relu.deinit();

    var negative_input = try Tensor.init(allocator, &.{4}, device);
    defer negative_input.deinit();
    try negative_input.setData(&.{ -2.0, -1.0, 1.0, 2.0 });

    std.debug.print("Input: ", .{});
    try negative_input.print();

    var relu_output = try relu.forward(&negative_input);
    defer relu_output.deinit();

    std.debug.print("ReLU Output: ", .{});
    try relu_output.print();
    std.debug.print("\n", .{});

    // Example 3: Complete MLP (Multi-Layer Perceptron)
    std.debug.print("🧠 Example 3: Complete MLP\n", .{});
    std.debug.print("---------------------------\n", .{});

    // Create MLP: input(2) -> hidden(3) -> output(1)
    var mlp = try nn.SimpleMLP.init(allocator, 2, 3, 1, device);
    defer mlp.deinit();

    var mlp_input = try Tensor.init(allocator, &.{2}, device);
    defer mlp_input.deinit();
    try mlp_input.setData(&.{ 0.5, -0.3 });

    std.debug.print("MLP Input: ", .{});
    try mlp_input.print();

    var mlp_output = try mlp.forward(&mlp_input);
    defer mlp_output.deinit();

    std.debug.print("MLP Output: ", .{});
    try mlp_output.print();
    std.debug.print("\n", .{});

    // Example 4: Batch Processing (2D input)
    std.debug.print("📦 Example 4: Batch Processing\n", .{});
    std.debug.print("-------------------------------\n", .{});

    var batch_linear = try nn.Linear.init(allocator, 2, 3, device, true);
    defer batch_linear.deinit();

    // Create batch input: 3 samples, each with 2 features
    var batch_input = try Tensor.init(allocator, &.{ 3, 2 }, device);
    defer batch_input.deinit();
    try batch_input.setData(&.{
        1.0, 2.0, // Sample 1
        -1.0, 0.5, // Sample 2
        0.0, -0.5, // Sample 3
    });

    std.debug.print("Batch Input (3 samples, 2 features each):\n", .{});
    try batch_input.print();

    var batch_output = try batch_linear.forward(&batch_input);
    defer batch_output.deinit();

    std.debug.print("Batch Output (3 samples, 3 features each):\n", .{});
    try batch_output.print();
    std.debug.print("\n", .{});

    // Example 5: Manual Forward Pass (step by step)
    std.debug.print("🔧 Example 5: Manual Forward Pass\n", .{});
    std.debug.print("----------------------------------\n", .{});

    // Create individual components
    var fc1 = try nn.Linear.init(allocator, 2, 4, device, true);
    defer fc1.deinit();

    var activation = nn.ReLU.init(allocator, device);
    defer activation.deinit();

    var fc2 = try nn.Linear.init(allocator, 4, 1, device, true);
    defer fc2.deinit();

    var sigmoid = nn.Sigmoid.init(allocator, device);
    defer sigmoid.deinit();

    // Manual forward pass
    var x = try Tensor.init(allocator, &.{2}, device);
    defer x.deinit();
    try x.setData(&.{ 1.5, -0.8 });

    std.debug.print("Step 1 - Input: ", .{});
    try x.print();

    // x = fc1(x)
    var x1 = try fc1.forward(&x);
    defer x1.deinit();
    std.debug.print("Step 2 - After FC1: ", .{});
    try x1.print();

    // x = relu(x)
    var x2 = try activation.forward(&x1);
    defer x2.deinit();
    std.debug.print("Step 3 - After ReLU: ", .{});
    try x2.print();

    // x = fc2(x)
    var x3 = try fc2.forward(&x2);
    defer x3.deinit();
    std.debug.print("Step 4 - After FC2: ", .{});
    try x3.print();

    // x = sigmoid(x)
    var final_output = try sigmoid.forward(&x3);
    defer final_output.deinit();
    std.debug.print("Step 5 - After Sigmoid: ", .{});
    try final_output.print();

    std.debug.print("\n✅ All PyTorch-like examples completed successfully!\n", .{});
    std.debug.print("\n🎯 Key Features Demonstrated:\n", .{});
    std.debug.print("   • Linear layers with bias\n", .{});
    std.debug.print("   • ReLU and Sigmoid activations\n", .{});
    std.debug.print("   • Multi-layer perceptrons\n", .{});
    std.debug.print("   • Batch processing (2D inputs)\n", .{});
    std.debug.print("   • Manual forward pass construction\n", .{});
    std.debug.print("   • PyTorch-like API design\n", .{});
}
