//! Automatic differentiation module for GradientZ
//! Contains gradient function structures and backward pass implementation

const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const zeros = @import("tensor.zig").zeros;
const ones = @import("tensor.zig").ones;

/// Gradient function interface - all backward functions implement this
pub const GradFn = struct {
    name: []const u8,
    inputs: std.ArrayList(*Tensor),
    backwardFn: *const fn (*GradFn, *Tensor) anyerror!void,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, name: []const u8, backward_fn: *const fn (*GradFn, *Tensor) anyerror!void) Self {
        return Self{
            .name = name,
            .inputs = std.ArrayList(*Tensor).init(allocator),
            .backwardFn = backward_fn,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.inputs.deinit();
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !void {
        return self.backwardFn(self, grad_output);
    }
};

/// Addition backward: both inputs get the same gradient
pub const AddBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "AddBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*AddBackward, @fieldParentPtr("grad_fn", grad_fn));
        const a = self.grad_fn.inputs.items[0];
        const b = self.grad_fn.inputs.items[1];

        // Addition: both inputs get the same gradient
        if (a.requires_grad) {
            if (a.grad == null) {
                a.grad = try a.allocator.create(Tensor);
                a.grad.?.* = try zeros(a.allocator, a.shape, a.device);
            }
            const temp = try a.grad.?.add(grad_output);
            a.grad.?.deinit();
            a.grad.?.* = temp;
        }

        if (b.requires_grad) {
            if (b.grad == null) {
                b.grad = try b.allocator.create(Tensor);
                b.grad.?.* = try zeros(b.allocator, b.shape, b.device);
            }
            const temp = try b.grad.?.add(grad_output);
            b.grad.?.deinit();
            b.grad.?.* = temp;
        }
    }
};

/// Subtraction backward: a gets gradient, b gets negative gradient
pub const SubBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "SubBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*SubBackward, @fieldParentPtr("grad_fn", grad_fn));
        const a = self.grad_fn.inputs.items[0];
        const b = self.grad_fn.inputs.items[1];

        // Subtraction: grad_a = grad_output, grad_b = -grad_output
        if (a.requires_grad) {
            if (a.grad == null) {
                a.grad = try a.allocator.create(Tensor);
                a.grad.?.* = try zeros(a.allocator, a.shape, a.device);
            }
            const temp = try a.grad.?.add(grad_output);
            a.grad.?.deinit();
            a.grad.?.* = temp;
        }

        if (b.requires_grad) {
            if (b.grad == null) {
                b.grad = try b.allocator.create(Tensor);
                b.grad.?.* = try zeros(b.allocator, b.shape, b.device);
            }
            var neg_grad = try grad_output.neg();
            defer neg_grad.deinit();
            const temp = try b.grad.?.add(&neg_grad);
            b.grad.?.deinit();
            b.grad.?.* = temp;
        }
    }
};

/// Multiplication backward
pub const MulBackward = struct {
    grad_fn: GradFn,
    a_data: *Tensor, // Store input values for gradient computation
    b_data: *Tensor,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "MulBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);

        // Store copies of input data for gradient computation
        self.a_data = try allocator.create(Tensor);
        self.a_data.* = try Tensor.init(allocator, a.shape, a.device);
        self.b_data = try allocator.create(Tensor);
        self.b_data.* = try Tensor.init(allocator, b.shape, b.device);

        const a_data = try @constCast(a).data();
        const b_data = try @constCast(b).data();
        const self_a_data = try self.a_data.data();
        const self_b_data = try self.b_data.data();

        @memcpy(self_a_data, a_data);
        @memcpy(self_b_data, b_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.a_data.deinit();
        self.grad_fn.allocator.destroy(self.a_data);
        self.b_data.deinit();
        self.grad_fn.allocator.destroy(self.b_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*MulBackward, @fieldParentPtr("grad_fn", grad_fn));
        const a = self.grad_fn.inputs.items[0];
        const b = self.grad_fn.inputs.items[1];

        // Multiplication: grad_a = grad_output * b, grad_b = grad_output * a
        if (a.requires_grad) {
            if (a.grad == null) {
                a.grad = try a.allocator.create(Tensor);
                a.grad.?.* = try zeros(a.allocator, a.shape, a.device);
            }
            var grad_a = try grad_output.mul(self.b_data);
            defer grad_a.deinit();
            const temp = try a.grad.?.add(&grad_a);
            a.grad.?.deinit();
            a.grad.?.* = temp;
        }

        if (b.requires_grad) {
            if (b.grad == null) {
                b.grad = try b.allocator.create(Tensor);
                b.grad.?.* = try zeros(b.allocator, b.shape, b.device);
            }
            var grad_b = try grad_output.mul(self.a_data);
            defer grad_b.deinit();
            const temp = try b.grad.?.add(&grad_b);
            b.grad.?.deinit();
            b.grad.?.* = temp;
        }
    }
};

/// Division backward: a / b
pub const DivBackward = struct {
    grad_fn: GradFn,
    a_data: *Tensor,
    b_data: *Tensor,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "DivBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);

        // Store copies of input data
        self.a_data = try allocator.create(Tensor);
        self.a_data.* = try Tensor.init(allocator, a.shape, a.device);
        self.b_data = try allocator.create(Tensor);
        self.b_data.* = try Tensor.init(allocator, b.shape, b.device);

        const a_data = try @constCast(a).data();
        const b_data = try @constCast(b).data();
        const self_a_data = try self.a_data.data();
        const self_b_data = try self.b_data.data();

        @memcpy(self_a_data, a_data);
        @memcpy(self_b_data, b_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.a_data.deinit();
        self.grad_fn.allocator.destroy(self.a_data);
        self.b_data.deinit();
        self.grad_fn.allocator.destroy(self.b_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*DivBackward, @fieldParentPtr("grad_fn", grad_fn));
        const a = self.grad_fn.inputs.items[0];
        const b = self.grad_fn.inputs.items[1];

        // Division: grad_a = grad_output / b, grad_b = -grad_output * a / (b * b)
        if (a.requires_grad) {
            if (a.grad == null) {
                a.grad = try a.allocator.create(Tensor);
                a.grad.?.* = try zeros(a.allocator, a.shape, a.device);
            }
            var grad_a = try grad_output.div(self.b_data);
            defer grad_a.deinit();
            const temp = try a.grad.?.add(&grad_a);
            a.grad.?.deinit();
            a.grad.?.* = temp;
        }

        if (b.requires_grad) {
            if (b.grad == null) {
                b.grad = try b.allocator.create(Tensor);
                b.grad.?.* = try zeros(b.allocator, b.shape, b.device);
            }
            var neg_grad = try grad_output.neg();
            defer neg_grad.deinit();
            var temp1 = try neg_grad.mul(self.a_data);
            defer temp1.deinit();
            var b_squared = try self.b_data.mul(self.b_data);
            defer b_squared.deinit();
            var grad_b = try temp1.div(&b_squared);
            defer grad_b.deinit();
            const temp2 = try b.grad.?.add(&grad_b);
            b.grad.?.deinit();
            b.grad.?.* = temp2;
        }
    }
};

/// Power backward: a^b (element-wise)
pub const PowBackward = struct {
    grad_fn: GradFn,
    a_data: *Tensor,
    b_data: *Tensor,
    result_data: *Tensor,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor, result: *const Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "PowBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);

        // Store copies of input and result data
        self.a_data = try allocator.create(Tensor);
        self.a_data.* = try Tensor.init(allocator, a.shape, a.device);
        self.b_data = try allocator.create(Tensor);
        self.b_data.* = try Tensor.init(allocator, b.shape, b.device);
        self.result_data = try allocator.create(Tensor);
        self.result_data.* = try Tensor.init(allocator, result.shape, result.device);

        const a_data = try @constCast(a).data();
        const b_data = try @constCast(b).data();
        const result_data = try @constCast(result).data();
        const self_a_data = try self.a_data.data();
        const self_b_data = try self.b_data.data();
        const self_result_data = try self.result_data.data();

        @memcpy(self_a_data, a_data);
        @memcpy(self_b_data, b_data);
        @memcpy(self_result_data, result_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.a_data.deinit();
        self.grad_fn.allocator.destroy(self.a_data);
        self.b_data.deinit();
        self.grad_fn.allocator.destroy(self.b_data);
        self.result_data.deinit();
        self.grad_fn.allocator.destroy(self.result_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*PowBackward, @fieldParentPtr("grad_fn", grad_fn));
        const a = self.grad_fn.inputs.items[0];
        const b = self.grad_fn.inputs.items[1];

        // Power: grad_a = grad_output * b * a^(b-1), grad_b = grad_output * a^b * ln(a)
        if (a.requires_grad) {
            if (a.grad == null) {
                a.grad = try a.allocator.create(Tensor);
                a.grad.?.* = try zeros(a.allocator, a.shape, a.device);
            }

            // Create b-1 tensor
            var ones_tensor = try ones(a.allocator, self.b_data.shape, self.b_data.device);
            defer ones_tensor.deinit();
            var b_minus_1 = try self.b_data.sub(&ones_tensor);
            defer b_minus_1.deinit();

            // a^(b-1)
            var a_pow_b_minus_1 = try self.a_data.pow(&b_minus_1);
            defer a_pow_b_minus_1.deinit();

            // grad_output * b * a^(b-1)
            var temp1 = try grad_output.mul(self.b_data);
            defer temp1.deinit();
            var grad_a = try temp1.mul(&a_pow_b_minus_1);
            defer grad_a.deinit();

            const temp2 = try a.grad.?.add(&grad_a);
            a.grad.?.deinit();
            a.grad.?.* = temp2;
        }

        if (b.requires_grad) {
            if (b.grad == null) {
                b.grad = try b.allocator.create(Tensor);
                b.grad.?.* = try zeros(b.allocator, b.shape, b.device);
            }

            // grad_output * a^b * ln(a)
            var ln_a = try self.a_data.log();
            defer ln_a.deinit();
            var temp1 = try grad_output.mul(self.result_data);
            defer temp1.deinit();
            var grad_b = try temp1.mul(&ln_a);
            defer grad_b.deinit();

            const temp2 = try b.grad.?.add(&grad_b);
            b.grad.?.deinit();
            b.grad.?.* = temp2;
        }
    }
};

/// ReLU backward
pub const ReluBackward = struct {
    grad_fn: GradFn,
    input_data: *Tensor,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "ReluBackward", backward);
        try self.grad_fn.inputs.append(input);

        // Store copy of input data
        self.input_data = try allocator.create(Tensor);
        self.input_data.* = try Tensor.init(allocator, input.shape, input.device);
        const input_data = try @constCast(input).data();
        const self_input_data = try self.input_data.data();
        @memcpy(self_input_data, input_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.input_data.deinit();
        self.grad_fn.allocator.destroy(self.input_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*ReluBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // Create ReLU mask (1 where input > 0, 0 otherwise)
            var relu_mask = try createReluMask(input.allocator, self.input_data);
            defer relu_mask.deinit();

            var grad_input = try grad_output.mul(&relu_mask);
            defer grad_input.deinit();

            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Negation backward
pub const NegBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "NegBackward", backward);
        try self.grad_fn.inputs.append(input);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*NegBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            var neg_grad = try grad_output.neg();
            defer neg_grad.deinit();
            const temp = try input.grad.?.add(&neg_grad);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Tanh backward
pub const TanhBackward = struct {
    grad_fn: GradFn,
    output_data: *Tensor, // Store tanh output for gradient computation

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor, output: *const Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "TanhBackward", backward);
        try self.grad_fn.inputs.append(input);

        // Store copy of output data (tanh values)
        self.output_data = try allocator.create(Tensor);
        self.output_data.* = try Tensor.init(allocator, output.shape, output.device);
        const output_data = try @constCast(output).data();
        const self_output_data = try self.output_data.data();
        @memcpy(self_output_data, output_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.output_data.deinit();
        self.grad_fn.allocator.destroy(self.output_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*TanhBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // tanh'(x) = 1 - tanh²(x)
            var ones_tensor = try ones(input.allocator, self.output_data.shape, self.output_data.device);
            defer ones_tensor.deinit();
            var tanh_squared = try self.output_data.mul(self.output_data);
            defer tanh_squared.deinit();
            var derivative = try ones_tensor.sub(&tanh_squared);
            defer derivative.deinit();

            var grad_input = try grad_output.mul(&derivative);
            defer grad_input.deinit();

            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Sigmoid backward
pub const SigmoidBackward = struct {
    grad_fn: GradFn,
    output_data: *Tensor, // Store sigmoid output

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor, output: *const Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "SigmoidBackward", backward);
        try self.grad_fn.inputs.append(input);

        // Store copy of output data (sigmoid values)
        self.output_data = try allocator.create(Tensor);
        self.output_data.* = try Tensor.init(allocator, output.shape, output.device);
        const output_data = try @constCast(output).data();
        const self_output_data = try self.output_data.data();
        @memcpy(self_output_data, output_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.output_data.deinit();
        self.grad_fn.allocator.destroy(self.output_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*SigmoidBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            var ones_tensor = try ones(input.allocator, self.output_data.shape, self.output_data.device);
            defer ones_tensor.deinit();
            var one_minus_sigmoid = try ones_tensor.sub(self.output_data);
            defer one_minus_sigmoid.deinit();
            var derivative = try self.output_data.mul(&one_minus_sigmoid);
            defer derivative.deinit();

            var grad_input = try grad_output.mul(&derivative);
            defer grad_input.deinit();

            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Exponential backward
pub const ExpBackward = struct {
    grad_fn: GradFn,
    output_data: *Tensor, // Store exp output

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor, output: *const Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "ExpBackward", backward);
        try self.grad_fn.inputs.append(input);

        // Store copy of output data (exp values)
        self.output_data = try allocator.create(Tensor);
        self.output_data.* = try Tensor.init(allocator, output.shape, output.device);
        const output_data = try @constCast(output).data();
        const self_output_data = try self.output_data.data();
        @memcpy(self_output_data, output_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.output_data.deinit();
        self.grad_fn.allocator.destroy(self.output_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*ExpBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // exp'(x) = exp(x)
            var grad_input = try grad_output.mul(self.output_data);
            defer grad_input.deinit();

            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Logarithm backward
pub const LogBackward = struct {
    grad_fn: GradFn,
    input_data: *Tensor,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "LogBackward", backward);
        try self.grad_fn.inputs.append(input);

        // Store copy of input data
        self.input_data = try allocator.create(Tensor);
        self.input_data.* = try Tensor.init(allocator, input.shape, input.device);
        const input_data = try @constCast(input).data();
        const self_input_data = try self.input_data.data();
        @memcpy(self_input_data, input_data);

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.input_data.deinit();
        self.grad_fn.allocator.destroy(self.input_data);
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*LogBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // log'(x) = 1/x
            var grad_input = try grad_output.div(self.input_data);
            defer grad_input.deinit();

            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Helper function to create ReLU mask (1 where input > 0, 0 otherwise)
fn createReluMask(allocator: std.mem.Allocator, input: *const Tensor) !Tensor {
    var result = try Tensor.init(allocator, input.shape, input.device);
    const input_data = try @constCast(input).data();
    const result_data = try result.data();

    for (0..input.num_elements) |i| {
        result_data[i] = if (input_data[i] > 0.0) 1.0 else 0.0;
    }
    return result;
}

/// Matrix multiplication backward pass
pub const MatmulBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "MatmulBackward", backward);
        try self.grad_fn.inputs.append(a);
        try self.grad_fn.inputs.append(b);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const a = grad_fn.inputs.items[0];
        const b = grad_fn.inputs.items[1];

        // For C = A * B:
        // dA = grad_output * B^T
        // dB = A^T * grad_output

        if (a.grad != null) {
            // Compute B transpose and multiply
            var b_t = try transpose(b);
            defer b_t.deinit();
            var grad_a = try grad_output.matmul(&b_t);
            defer grad_a.deinit();

            // Add to existing gradient
            const temp = try a.grad.?.add(&grad_a);
            a.grad.?.deinit();
            a.grad.?.* = temp;
        }

        if (b.grad != null) {
            // Compute A transpose and multiply
            var a_t = try transpose(a);
            defer a_t.deinit();
            var grad_b = try a_t.matmul(grad_output);
            defer grad_b.deinit();

            // Add to existing gradient
            const temp = try b.grad.?.add(&grad_b);
            b.grad.?.deinit();
            b.grad.?.* = temp;
        }
    }
};

/// Mean backward pass
pub const MeanBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "MeanBackward", backward);
        try self.grad_fn.inputs.append(input);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const input = grad_fn.inputs.items[0];

        if (input.grad != null) {
            // Mean gradient is broadcast: grad_input = grad_output / num_elements
            const num_elements = @as(f32, @floatFromInt(input.num_elements));
            const grad_scalar = try grad_output.constData();
            const grad_value = grad_scalar[0] / num_elements;

            // Create gradient tensor filled with the gradient value
            var grad_input = try Tensor.init(grad_fn.allocator, input.shape, input.device);
            try grad_input.fill(grad_value);
            defer grad_input.deinit();

            // Add to existing gradient
            const temp = try input.grad.?.add(&grad_input);
            input.grad.?.deinit();
            input.grad.?.* = temp;
        }
    }
};

/// Helper function to transpose a 2D tensor
fn transpose(tensor: *Tensor) !Tensor {
    if (tensor.shape.len != 2) return error.InvalidShape;

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];

    var result = try Tensor.init(tensor.allocator, &.{ cols, rows }, tensor.device);
    const input_data = try tensor.constData();
    const output_data = try result.data();

    for (0..rows) |i| {
        for (0..cols) |j| {
            output_data[j * rows + i] = input_data[i * cols + j];
        }
    }

    return result;
}

/// Sum backward: gradient is broadcast to all input elements
pub const SumBackward = struct {
    grad_fn: GradFn,

    const Self = @This();

    pub fn create(allocator: std.mem.Allocator, input: *Tensor) !*Self {
        var self = try allocator.create(Self);
        self.grad_fn = GradFn.init(allocator, "SumBackward", backward);
        try self.grad_fn.inputs.append(input);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.grad_fn.deinit();
        self.grad_fn.allocator.destroy(self);
    }

    fn backward(grad_fn: *GradFn, grad_output: *Tensor) !void {
        const self = @as(*SumBackward, @fieldParentPtr("grad_fn", grad_fn));
        const input = self.grad_fn.inputs.items[0];

        // Sum backward: gradient is broadcast to all input elements
        if (input.requires_grad) {
            if (input.grad == null) {
                input.grad = try input.allocator.create(Tensor);
                input.grad.?.* = try zeros(input.allocator, input.shape, input.device);
            }

            // Get the scalar gradient value and broadcast it to all input elements
            const grad_scalar = (try grad_output.constData())[0];
            const input_grad_data = try input.grad.?.data();
            for (input_grad_data) |*elem| {
                elem.* += grad_scalar;
            }
        }
    }
};
