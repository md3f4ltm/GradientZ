const std = @import("std");
const lib = @import("../lib.zig");
const Tensor = lib.Tensor;
const Device = lib.Device;

/// Base Module interface - similar to PyTorch's nn.Module
pub const Module = struct {
    allocator: std.mem.Allocator,
    device: Device,
    training: bool = true,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, device: Device) Self {
        return Self{
            .allocator = allocator,
            .device = device,
            .training = true,
        };
    }

    pub fn train(self: *Self) void {
        self.training = true;
    }

    pub fn eval(self: *Self) void {
        self.training = false;
    }
};

/// Linear layer - equivalent to PyTorch's nn.Linear
pub const Linear = struct {
    module: Module,
    weight: Tensor,
    bias: ?Tensor,
    in_features: usize,
    out_features: usize,

    const Self = @This();

    /// Initialize a Linear layer
    /// in_features: input dimension
    /// out_features: output dimension
    /// bias: whether to include bias term
    pub fn init(allocator: std.mem.Allocator, in_features: usize, out_features: usize, device: Device, bias: bool) !Self {
        var linear = Self{
            .module = Module.init(allocator, device),
            .weight = undefined,
            .bias = null,
            .in_features = in_features,
            .out_features = out_features,
        };

        // Initialize weight matrix (out_features x in_features)
        linear.weight = try Tensor.initWithGrad(allocator, &.{ out_features, in_features }, device, true);

        // Xavier/Glorot initialization for weights
        const fan_in = @as(f32, @floatFromInt(in_features));
        const fan_out = @as(f32, @floatFromInt(out_features));
        const std_dev = @sqrt(2.0 / (fan_in + fan_out));

        // Simple random initialization (in real implementation, use proper random)
        const weight_data = try allocator.alloc(f32, in_features * out_features);
        defer allocator.free(weight_data);

        var prng = std.Random.DefaultPrng.init(42);
        const random = prng.random();

        for (weight_data, 0..) |*val, i| {
            _ = i;
            val.* = (random.float(f32) - 0.5) * 2.0 * std_dev;
        }
        try linear.weight.setData(weight_data);

        // Initialize bias if requested
        if (bias) {
            linear.bias = try Tensor.initWithGrad(allocator, &.{out_features}, device, true);
            const bias_data = try allocator.alloc(f32, out_features);
            defer allocator.free(bias_data);

            for (bias_data) |*val| {
                val.* = 0.0; // Initialize bias to zero
            }
            try linear.bias.?.setData(bias_data);
        }

        return linear;
    }

    /// Forward pass through linear layer
    /// input: tensor of shape (batch_size, in_features) or (in_features,)
    /// returns: tensor of shape (batch_size, out_features) or (out_features,)
    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        // Handle both 1D and 2D inputs
        const input_shape = input.shape;

        var output: Tensor = undefined;

        if (input_shape.len == 1) {
            // 1D input: (in_features,) -> (out_features,)
            if (input_shape[0] != self.in_features) {
                return error.ShapeMismatch;
            }

            // Create a temporary 2D tensor for matrix multiplication
            var temp_input = try Tensor.init(self.module.allocator, &.{ 1, self.in_features }, self.module.device);
            defer temp_input.deinit();

            // Copy data from 1D input to 2D temp tensor
            const input_data = try input.constData();
            try temp_input.setData(input_data);

            // output = temp_input @ weight.T
            var weight_t = try self.weight.transpose();
            defer weight_t.deinit();

            var temp_output = try temp_input.matmul(&weight_t);
            defer temp_output.deinit();

            // Create 1D output tensor with gradient tracking
            output = try Tensor.initWithGrad(self.module.allocator, &.{self.out_features}, self.module.device, temp_output.requires_grad);
            const temp_data = try temp_output.constData();
            try output.setData(temp_data);
        } else if (input_shape.len == 2) {
            // 2D input: (batch_size, in_features) -> (batch_size, out_features)
            if (input_shape[1] != self.in_features) {
                return error.ShapeMismatch;
            }

            // output = input @ weight.T
            var weight_t = try self.weight.transpose();
            defer weight_t.deinit();

            output = try input.matmul(&weight_t);
        } else {
            return error.UnsupportedInputShape;
        }

        // Add bias if present
        if (self.bias) |*bias| {
            if (input_shape.len == 1) {
                // For 1D input, bias can be added directly
                const output_with_bias = try output.add(bias);
                output.deinit();
                output = output_with_bias;
            } else {
                // For 2D input (batch), need to broadcast bias across batch dimension
                // Create a temporary bias tensor with the same shape as output
                const batch_size = input_shape[0];
                var broadcasted_bias = try Tensor.init(self.module.allocator, &.{ batch_size, self.out_features }, self.module.device);
                defer broadcasted_bias.deinit();

                // Fill the broadcasted bias with repeated bias values
                const bias_data = try bias.constData();
                const broadcast_data = try broadcasted_bias.data();

                for (0..batch_size) |batch_idx| {
                    for (0..self.out_features) |feature_idx| {
                        broadcast_data[batch_idx * self.out_features + feature_idx] = bias_data[feature_idx];
                    }
                }

                const output_with_bias = try output.add(&broadcasted_bias);
                output.deinit();
                output = output_with_bias;
            }
        }

        return output;
    }

    pub fn deinit(self: *Self) void {
        self.weight.deinit();
        if (self.bias) |*bias| {
            bias.deinit();
        }
    }
};

/// ReLU activation function
pub const ReLU = struct {
    module: Module,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, device: Device) Self {
        return Self{
            .module = Module.init(allocator, device),
        };
    }

    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        _ = self;
        return try input.relu();
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        // No cleanup needed for ReLU
    }
};

/// Sigmoid activation function
pub const Sigmoid = struct {
    module: Module,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, device: Device) Self {
        return Self{
            .module = Module.init(allocator, device),
        };
    }

    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        _ = self;
        return try input.sigmoid();
    }

    pub fn deinit(self: *Self) void {
        _ = self;
        // No cleanup needed for Sigmoid
    }
};

/// Simple Multi-Layer Perceptron (MLP) - equivalent to the PyTorch example
pub const SimpleMLP = struct {
    module: Module,
    fc1: Linear,
    fc2: Linear,
    relu: ReLU,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize, device: Device) !Self {
        return Self{
            .module = Module.init(allocator, device),
            .fc1 = try Linear.init(allocator, input_size, hidden_size, device, true),
            .fc2 = try Linear.init(allocator, hidden_size, output_size, device, true),
            .relu = ReLU.init(allocator, device),
        };
    }

    pub fn forward(self: *Self, input: *const Tensor) !Tensor {
        // x = fc1(input)
        var x = try self.fc1.forward(input);

        // x = relu(x)
        var x_relu = try self.relu.forward(&x);
        x.deinit();

        // x = fc2(x)
        const output = try self.fc2.forward(&x_relu);
        x_relu.deinit();

        return output;
    }

    pub fn deinit(self: *Self) void {
        self.fc1.deinit();
        self.fc2.deinit();
        self.relu.deinit();
    }
};

/// Test function for the neural network module
pub fn testNN(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧠 Testing Neural Network Module...\n", .{});

    const device = lib.cpu();

    // Test Linear layer
    std.debug.print("   Testing Linear layer...\n", .{});
    var linear = try Linear.init(allocator, 3, 2, device, true);
    defer linear.deinit();

    // Create test input
    var input = try Tensor.init(allocator, &.{3}, device);
    defer input.deinit();
    try input.setData(&.{ 1.0, 2.0, 3.0 });

    std.debug.print("     Input: ", .{});
    try input.print();

    var output = try linear.forward(&input);
    defer output.deinit();

    std.debug.print("     Linear output: ", .{});
    try output.print();

    // Test ReLU
    std.debug.print("   Testing ReLU activation...\n", .{});
    var relu = ReLU.init(allocator, device);
    defer relu.deinit();

    var negative_input = try Tensor.init(allocator, &.{4}, device);
    defer negative_input.deinit();
    try negative_input.setData(&.{ -2.0, -1.0, 1.0, 2.0 });

    std.debug.print("     Input: ", .{});
    try negative_input.print();

    var relu_output = try relu.forward(&negative_input);
    defer relu_output.deinit();

    std.debug.print("     ReLU output: ", .{});
    try relu_output.print();

    // Test SimpleMLP
    std.debug.print("   Testing SimpleMLP...\n", .{});
    var mlp = try SimpleMLP.init(allocator, 2, 3, 1, device);
    defer mlp.deinit();

    var mlp_input = try Tensor.init(allocator, &.{2}, device);
    defer mlp_input.deinit();
    try mlp_input.setData(&.{ 0.5, -0.3 });

    std.debug.print("     MLP Input: ", .{});
    try mlp_input.print();

    var mlp_output = try mlp.forward(&mlp_input);
    defer mlp_output.deinit();

    std.debug.print("     MLP Output: ", .{});
    try mlp_output.print();

    std.debug.print("   ✅ Neural Network module test completed!\n", .{});
}
