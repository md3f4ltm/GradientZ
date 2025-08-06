const std = @import("std");
const GradientZ = @import("GradientZ_lib");
const Tensor = GradientZ.Tensor;
const Device = GradientZ.Device;

pub fn main() !void {
    // Initialize the library
    GradientZ.init();

    // Create allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Autograd (Automatic Differentiation) Example ===\n\n", .{});

    try testAutograd(allocator);

    std.debug.print("\n✅ Autograd example completed successfully!\n", .{});
}

fn testAutograd(allocator: std.mem.Allocator) !void {
    // Test 1: Simple addition and multiplication
    std.debug.print("Test 1: Simple operations (a + b) * c\n", .{});

    var a = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer a.deinit();
    try a.setData(&.{ 1.0, 2.0, 3.0, 4.0 });

    var b = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer b.deinit();
    try b.setData(&.{ 5.0, 6.0, 7.0, 8.0 });

    var c = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer c.deinit();
    try c.setData(&.{ 2.0, 2.0, 2.0, 2.0 });

    std.debug.print("a = \n", .{});
    try a.print();
    std.debug.print("b = \n", .{});
    try b.print();
    std.debug.print("c = \n", .{});
    try c.print();

    // Forward pass: result = (a + b) * c
    var sum = try a.add(&b);
    defer sum.deinit();
    var result = try sum.mul(&c);
    defer result.deinit();

    std.debug.print("result = (a + b) * c:\n", .{});
    try result.print();

    // Backward pass
    try result.backward();

    std.debug.print("Gradients after backward():\n", .{});
    if (a.grad) |grad_a| {
        std.debug.print("grad_a:\n", .{});
        try grad_a.print();
    }
    if (b.grad) |grad_b| {
        std.debug.print("grad_b:\n", .{});
        try grad_b.print();
    }
    if (c.grad) |grad_c| {
        std.debug.print("grad_c:\n", .{});
        try grad_c.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 2: Power and subtraction
    std.debug.print("Test 2: Power operation x^2 - y\n", .{});

    var x = try Tensor.initWithGrad(allocator, &.{ 1, 3 }, Device.cpu(), true);
    defer x.deinit();
    try x.setData(&.{ 2.0, 3.0, -1.0 });

    var y = try Tensor.initWithGrad(allocator, &.{ 1, 3 }, Device.cpu(), true);
    defer y.deinit();
    try y.setData(&.{ 1.0, 4.0, 2.0 });

    var two = try Tensor.init(allocator, &.{ 1, 3 }, Device.cpu());
    defer two.deinit();
    try two.setData(&.{ 2.0, 2.0, 2.0 });

    std.debug.print("x = \n", .{});
    try x.print();
    std.debug.print("y = \n", .{});
    try y.print();

    // Forward pass: result = x^2 - y
    var x_squared = try x.pow(&two);
    defer x_squared.deinit();
    var result2 = try x_squared.sub(&y);
    defer result2.deinit();

    std.debug.print("result = x^2 - y:\n", .{});
    try result2.print();

    // Backward pass
    try result2.backward();

    std.debug.print("Gradients:\n", .{});
    if (x.grad) |grad_x| {
        std.debug.print("grad_x (should be 2*x = [4, 6, -2]):\n", .{});
        try grad_x.print();
    }
    if (y.grad) |grad_y| {
        std.debug.print("grad_y (should be [-1, -1, -1]):\n", .{});
        try grad_y.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 3: Activation functions
    std.debug.print("Test 3: ReLU activation\n", .{});

    var input = try Tensor.initWithGrad(allocator, &.{ 1, 4 }, Device.cpu(), true);
    defer input.deinit();
    try input.setData(&.{ -2.0, -1.0, 1.0, 2.0 });

    std.debug.print("input = \n", .{});
    try input.print();

    // Forward pass: ReLU
    var relu_result = try input.relu();
    defer relu_result.deinit();

    std.debug.print("ReLU(input):\n", .{});
    try relu_result.print();

    // Backward pass
    try relu_result.backward();

    std.debug.print("Gradients:\n", .{});
    if (input.grad) |grad_input| {
        std.debug.print("grad_input (should be [0, 0, 1, 1]):\n", .{});
        try grad_input.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 4: Tanh activation
    std.debug.print("Test 4: Tanh activation\n", .{});

    var tanh_input = try Tensor.initWithGrad(allocator, &.{ 1, 3 }, Device.cpu(), true);
    defer tanh_input.deinit();
    try tanh_input.setData(&.{ 0.0, 1.0, -1.0 });

    std.debug.print("input = \n", .{});
    try tanh_input.print();

    // Forward pass: tanh
    var tanh_result = try tanh_input.tanh();
    defer tanh_result.deinit();

    std.debug.print("tanh(input):\n", .{});
    try tanh_result.print();

    // Backward pass
    try tanh_result.backward();

    std.debug.print("Gradients:\n", .{});
    if (tanh_input.grad) |grad_tanh| {
        std.debug.print("grad_input (tanh derivative: 1 - tanh^2):\n", .{});
        try grad_tanh.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 5: Chain rule with multiple operations
    std.debug.print("Test 5: Complex chain rule: sigmoid(x * w + b)\n", .{});

    var weight = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer weight.deinit();
    try weight.setData(&.{ 0.5, -0.3, 0.2, 0.7 });

    var bias = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer bias.deinit();
    try bias.setData(&.{ 0.1, -0.1, 0.0, 0.2 });

    var x_input = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer x_input.deinit();
    try x_input.setData(&.{ 1.0, 2.0, -1.0, 0.5 });

    std.debug.print("weight = \n", .{});
    try weight.print();
    std.debug.print("bias = \n", .{});
    try bias.print();
    std.debug.print("x = \n", .{});
    try x_input.print();

    // Forward pass: sigmoid(x * w + b)
    var wx = try x_input.mul(&weight);
    defer wx.deinit();
    var wx_plus_b = try wx.add(&bias);
    defer wx_plus_b.deinit();
    var output = try wx_plus_b.sigmoid();
    defer output.deinit();

    std.debug.print("output = sigmoid(x * w + b):\n", .{});
    try output.print();

    // Backward pass
    try output.backward();

    std.debug.print("Gradients:\n", .{});
    if (weight.grad) |grad_w| {
        std.debug.print("grad_weight:\n", .{});
        try grad_w.print();
    }
    if (bias.grad) |grad_b| {
        std.debug.print("grad_bias:\n", .{});
        try grad_b.print();
    }
    if (x_input.grad) |grad_x| {
        std.debug.print("grad_x:\n", .{});
        try grad_x.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 6: Gradient accumulation
    std.debug.print("Test 6: Gradient accumulation (y = x + x)\n", .{});

    var x_accum = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer x_accum.deinit();
    try x_accum.setData(&.{ 1.0, 2.0, 3.0, 4.0 });

    std.debug.print("x = \n", .{});
    try x_accum.print();

    // Forward pass: y = x + x (x is used twice)
    var y_accum = try x_accum.add(&x_accum);
    defer y_accum.deinit();

    std.debug.print("y = x + x:\n", .{});
    try y_accum.print();

    // Backward pass
    try y_accum.backward();

    std.debug.print("Gradients:\n", .{});
    if (x_accum.grad) |grad_x_accum| {
        std.debug.print("grad_x (should be [2, 2, 2, 2] due to accumulation):\n", .{});
        try grad_x_accum.print();
    }

    std.debug.print("\n" ++ "=" ** 50 ++ "\n\n", .{});

    // Test 7: Simple neural network layer
    std.debug.print("Test 7: Simple neural network forward and backward\n", .{});

    // Create a simple 2-input, 2-output linear layer with tanh activation
    var w1 = try Tensor.initWithGrad(allocator, &.{ 2, 2 }, Device.cpu(), true);
    defer w1.deinit();
    try w1.setData(&.{ 0.1, 0.2, 0.3, 0.4 }); // Initialize with small values

    var b1 = try Tensor.initWithGrad(allocator, &.{2}, Device.cpu(), true);
    defer b1.deinit();
    try b1.setData(&.{ 0.1, -0.1 });

    var input_data = try Tensor.init(allocator, &.{2}, Device.cpu());
    defer input_data.deinit();
    try input_data.setData(&.{ 1.0, -0.5 });

    std.debug.print("Input: [1.0, -0.5]\n", .{});
    std.debug.print("Weight matrix:\n", .{});
    try w1.print();
    std.debug.print("Bias vector:\n", .{});
    try b1.print();

    // Forward pass: output = tanh(input @ weight + bias)
    // For simplicity, we'll do element-wise operations
    var x1 = try Tensor.initWithGrad(allocator, &.{2}, Device.cpu(), false);
    defer x1.deinit();
    try x1.setData(&.{ 1.0, -0.5 });

    // In a real implementation, you'd use matrix multiplication
    // For this demo, we'll just show the setup

    std.debug.print("\n=== Autograd Demo completed successfully! ===\n", .{});
    std.debug.print("\nKey takeaways:\n", .{});
    std.debug.print("1. Gradients are computed automatically using the chain rule\n", .{});
    std.debug.print("2. Operations create computation graphs with backward functions\n", .{});
    std.debug.print("3. Gradients accumulate when variables are used multiple times\n", .{});
    std.debug.print("4. The backward() method propagates gradients through the graph\n", .{});
    std.debug.print("5. All mathematical operations support automatic differentiation\n", .{});
}
