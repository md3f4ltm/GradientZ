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

    std.debug.print("=== Training Example (Gradient Descent) ===\n\n", .{});

    try testMLPTraining(allocator);

    std.debug.print("\n✅ Training example completed successfully!\n", .{});
}

fn testMLPTraining(allocator: std.mem.Allocator) !void {
    std.debug.print("🎓 Testing Manual Linear Layer Training with .backward()...\n", .{});

    const device = GradientZ.cpu();

    // Create simple trainable parameters for a linear transformation: y = weight * x + bias
    var weight = try Tensor.initWithGrad(allocator, &.{1}, device, true);
    defer weight.deinit();
    try weight.setData(&.{1.0}); // Initialize to 1.0

    var bias = try Tensor.initWithGrad(allocator, &.{1}, device, true);
    defer bias.deinit();
    try bias.setData(&.{0.0}); // Initialize to 0.0

    std.debug.print("   Simple linear model: y = weight * x + bias\n", .{});
    std.debug.print("   Task: Learn to approximate y = 2*x (so weight should become 2, bias should stay 0)\n", .{});

    // Training parameters
    const learning_rate = 0.05;
    const epochs = 100;

    std.debug.print("   Starting training for {} epochs with learning rate {}...\n", .{ epochs, learning_rate });

    // Training data: simple linear relationship y = 2*x
    const training_inputs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const training_targets = [_]f32{ 2.0, 4.0, 6.0, 8.0 }; // y = 2*x

    // Training loop
    for (0..epochs) |epoch| {
        var total_loss: f32 = 0.0;

        // Train on each sample
        for (training_inputs, training_targets) |input_val, target_val| {
            // Create input and target tensors
            var input = try Tensor.init(allocator, &.{1}, device);
            defer input.deinit();
            try input.setData(&.{input_val});

            var target = try Tensor.init(allocator, &.{1}, device);
            defer target.deinit();
            try target.setData(&.{target_val});

            // Forward pass: y = weight * x + bias
            var weighted = try input.mul(&weight);
            defer weighted.deinit();

            var output = try weighted.add(&bias);
            defer output.deinit();

            // Compute loss (MSE)
            var diff = try output.sub(&target);
            defer diff.deinit();
            var loss = try diff.mul(&diff);
            defer loss.deinit();

            total_loss += (try loss.constData())[0];

            // Backward pass
            try loss.backward();

            // Update weight
            if (weight.grad) |grad| {
                const w_data = try weight.data();
                const g_data = try grad.constData();
                w_data[0] -= learning_rate * g_data[0];
                try grad.fill(0.0);
            }

            // Update bias
            if (bias.grad) |grad| {
                const b_data = try bias.data();
                const g_data = try grad.constData();
                b_data[0] -= learning_rate * g_data[0];
                try grad.fill(0.0);
            }
        }

        // Print progress
        if (epoch % 5 == 0) {
            const w_val = (try weight.constData())[0];
            const b_val = (try bias.constData())[0];
            std.debug.print("     Epoch {}: Loss = {:.4}, Weight = {:.4}, Bias = {:.4}\n", .{ epoch, total_loss / training_inputs.len, w_val, b_val });
        }
    }

    // Test final results
    std.debug.print("   Testing trained linear model:\n", .{});
    const final_weight = (try weight.constData())[0];
    const final_bias = (try bias.constData())[0];

    for (training_inputs, training_targets) |input_val, target_val| {
        var test_input = try Tensor.init(allocator, &.{1}, device);
        defer test_input.deinit();
        try test_input.setData(&.{input_val});

        // Forward pass
        var weighted = try test_input.mul(&weight);
        defer weighted.deinit();

        var output = try weighted.add(&bias);
        defer output.deinit();

        const predicted = (try output.constData())[0];
        const err = @abs(predicted - target_val);

        std.debug.print("     Input={:.1}: predicted={:.2}, target={:.1}, error={:.2}\n", .{ input_val, predicted, target_val, err });
    }

    std.debug.print("   Final parameters: Weight={:.4} (target: 2.0), Bias={:.4} (target: 0.0)\n", .{ final_weight, final_bias });

    // Check success
    const weight_error = @abs(final_weight - 2.0);
    const bias_error = @abs(final_bias - 0.0);

    if (weight_error < 0.1 and bias_error < 0.1) {
        std.debug.print("   ✅ Training successful! Learned linear relationship using .backward()\n", .{});
    } else {
        std.debug.print("   📈 Training improved: moving toward target parameters\n", .{});
    }

    std.debug.print("   🧠 Manual linear layer with .backward() demonstration completed!\n", .{});
}
