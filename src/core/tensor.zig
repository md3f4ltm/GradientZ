const std = @import("std");
const Device = @import("device.zig").Device;
const shape_utils = @import("shape.zig");
const backend_mod = @import("../backend/backend.zig");
const Backend = backend_mod.Backend;
const BackendMemory = backend_mod.BackendMemory;
const BackendError = backend_mod.BackendError;
const autograd = @import("autograd.zig");

pub const TensorError = error{
    ShapeMismatch,
    DeviceMismatch,
    InvalidShape,
    BackendError,
    OutOfMemory,
};

/// Tensor struct - only supports f32 for now
pub const Tensor = struct {
    allocator: std.mem.Allocator,
    shape: []usize,
    device: Device,
    memory: BackendMemory,
    num_elements: usize,

    // Gradient tracking fields
    grad: ?*Tensor, // Gradient tensor (optional)
    requires_grad: bool, // Whether to compute gradients
    grad_fn: ?*autograd.GradFn, // Backward function pointer

    const Self = @This();

    /// Create a new tensor with given shape on specified device and optional gradient tracking
    pub fn init(allocator: std.mem.Allocator, shape: []const usize, device: Device) !Self {
        return initWithGrad(allocator, shape, device, false);
    }
    // TODO: initWithGrad should be called onlu if the tensor requires gradients, not always.
    /// Create a new tensor with gradient tracking control
    pub fn initWithGrad(allocator: std.mem.Allocator, shape: []const usize, device: Device, requires_grad: bool) !Self {
        // Calculate number of elements
        const num_elements = shape_utils.calcElements(shape) catch |err| switch (err) {
            shape_utils.ShapeError.InvalidShape, shape_utils.ShapeError.EmptyShape => return TensorError.InvalidShape,
            else => return TensorError.InvalidShape,
        };

        // Get backend for device
        const backend = backend_mod.getBackend(device) catch return TensorError.BackendError;

        // Allocate memory
        const bytes_needed = num_elements * @sizeOf(f32);
        const memory = backend.alloc(allocator, device, bytes_needed) catch |err| switch (err) {
            BackendError.OutOfMemory => return TensorError.OutOfMemory,
            else => return TensorError.BackendError,
        };

        // Copy shape
        const shape_copy = shape_utils.copyShape(allocator, shape) catch return TensorError.OutOfMemory;

        return Self{
            .allocator = allocator,
            .shape = shape_copy,
            .device = device,
            .memory = memory,
            .num_elements = num_elements,
            .grad = null,
            .requires_grad = requires_grad,
            .grad_fn = null,
        };
    }

    /// Free tensor memory
    pub fn deinit(self: *Self) void {
        const backend = backend_mod.getBackend(self.device) catch return;
        backend.free(self.allocator, self.memory);
        self.allocator.free(self.shape);

        // Clean up gradient
        if (self.grad) |grad| {
            grad.deinit();
            self.allocator.destroy(grad);
        }

        // Clean up gradient function
        if (self.grad_fn) |grad_fn| {
            // Call the appropriate deinit method based on the grad_fn type
            // We need to determine which backward struct this is and call its deinit
            const grad_fn_name = grad_fn.name;
            if (std.mem.eql(u8, grad_fn_name, "AddBackward")) {
                const add_backward = @as(*autograd.AddBackward, @fieldParentPtr("grad_fn", grad_fn));
                add_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "SubBackward")) {
                const sub_backward = @as(*autograd.SubBackward, @fieldParentPtr("grad_fn", grad_fn));
                sub_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "MulBackward")) {
                const mul_backward = @as(*autograd.MulBackward, @fieldParentPtr("grad_fn", grad_fn));
                mul_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "DivBackward")) {
                const div_backward = @as(*autograd.DivBackward, @fieldParentPtr("grad_fn", grad_fn));
                div_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "PowBackward")) {
                const pow_backward = @as(*autograd.PowBackward, @fieldParentPtr("grad_fn", grad_fn));
                pow_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "ReluBackward")) {
                const relu_backward = @as(*autograd.ReluBackward, @fieldParentPtr("grad_fn", grad_fn));
                relu_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "NegBackward")) {
                const neg_backward = @as(*autograd.NegBackward, @fieldParentPtr("grad_fn", grad_fn));
                neg_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "TanhBackward")) {
                const tanh_backward = @as(*autograd.TanhBackward, @fieldParentPtr("grad_fn", grad_fn));
                tanh_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "SigmoidBackward")) {
                const sigmoid_backward = @as(*autograd.SigmoidBackward, @fieldParentPtr("grad_fn", grad_fn));
                sigmoid_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "ExpBackward")) {
                const exp_backward = @as(*autograd.ExpBackward, @fieldParentPtr("grad_fn", grad_fn));
                exp_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "LogBackward")) {
                const log_backward = @as(*autograd.LogBackward, @fieldParentPtr("grad_fn", grad_fn));
                log_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "MatmulBackward")) {
                const matmul_backward = @as(*autograd.MatmulBackward, @fieldParentPtr("grad_fn", grad_fn));
                matmul_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "MeanBackward")) {
                const mean_backward = @as(*autograd.MeanBackward, @fieldParentPtr("grad_fn", grad_fn));
                mean_backward.deinit();
            } else if (std.mem.eql(u8, grad_fn_name, "SumBackward")) {
                const sum_backward = @as(*autograd.SumBackward, @fieldParentPtr("grad_fn", grad_fn));
                sum_backward.deinit();
            }
        }

        self.* = undefined;
    }

    /// Get tensor data as f32 slice (CPU only for now)
    pub fn data(self: *Self) ![]f32 {
        if (!self.device.isCpu()) return TensorError.DeviceMismatch;
        return self.memory.asType(f32)[0..self.num_elements];
    }

    /// Get tensor data as const f32 slice (CPU only for now)
    pub fn constData(self: *const Self) ![]const f32 {
        if (!self.device.isCpu()) return TensorError.DeviceMismatch;
        return self.memory.asType(f32)[0..self.num_elements];
    }

    /// Fill tensor with a constant value
    pub fn fill(self: *Self, value: f32) !void {
        const tensor_data = try self.data();
        for (tensor_data) |*elem| {
            elem.* = value;
        }
    }

    /// Set tensor data from slice
    pub fn setData(self: *Self, values: []const f32) !void {
        if (values.len != self.num_elements) return TensorError.ShapeMismatch;
        const tensor_data = try self.data();
        @memcpy(tensor_data, values);
    }

    /// Add two tensors element-wise
    pub fn add(self: *const Self, other: *const Self) !Self {
        // Check compatibility
        if (!shape_utils.shapesEqual(self.shape, other.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }

        // Create result tensor
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad or other.requires_grad;

        // Perform addition using backend
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.add(result.memory, self.memory, other.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const add_backward = autograd.AddBackward.create(self.allocator, @constCast(self), @constCast(other)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &add_backward.grad_fn;
        }

        return result;
    }

    /// Matrix multiplication: self × other
    /// self: [m, k], other: [k, n] -> result: [m, n]
    pub fn matmul(self: *const Self, other: *const Self) !Self {
        // Check that tensors are 2D
        if (self.shape.len != 2 or other.shape.len != 2) {
            return TensorError.ShapeMismatch;
        }

        const m = self.shape[0];
        const k = self.shape[1];
        const k2 = other.shape[0];
        const n = other.shape[1];

        // Check dimension compatibility
        if (k != k2) {
            return TensorError.ShapeMismatch;
        }

        // Check device compatibility
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }

        // Create result tensor [m, n]
        var result = try Self.initWithGrad(self.allocator, &.{ m, n }, self.device, self.requires_grad or other.requires_grad);

        // Perform matrix multiplication using backend
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.matmul(result.memory, self.memory, other.memory, m, k, n) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Set up gradient function if needed
        if (result.requires_grad) {
            const matmul_backward = autograd.MatmulBackward.create(self.allocator, @constCast(self), @constCast(other)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &matmul_backward.grad_fn;
        }

        return result;
    }

    /// Element-wise division: self / other
    pub fn div(self: *const Self, other: *const Self) !Self {
        // Check compatibility
        if (!shape_utils.shapesEqual(self.shape, other.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad or other.requires_grad;

        // Perform division using backend
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.div(result.memory, self.memory, other.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const div_backward = autograd.DivBackward.create(self.allocator, @constCast(self), @constCast(other)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &div_backward.grad_fn;
        }

        return result;
    }
    /// Element-wise multiplication: self * other
    pub fn mul(self: *const Self, other: *const Self) !Self {
        if (!shape_utils.shapesEqual(self.shape, other.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }

        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad or other.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.DeviceMismatch;
        };

        backend.mul(result.memory, self.memory, other.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const mul_backward = autograd.MulBackward.create(self.allocator, @constCast(self), @constCast(other)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &mul_backward.grad_fn;
        }

        return result;
    }

    /// Element-wise power: self ^ other
    pub fn pow(self: *const Self, other: *const Self) !Self {
        // Check compatibility
        if (!shape_utils.shapesEqual(self.shape, other.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }

        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad or other.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.pow(result.memory, self.memory, other.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const pow_backward = autograd.PowBackward.create(self.allocator, @constCast(self), @constCast(other), &result) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &pow_backward.grad_fn;
        }

        return result;
    }

    /// Element-wise subtraction: self - other
    pub fn sub(self: *const Self, other: *const Self) !Self {
        // Check compatibility
        if (!shape_utils.shapesEqual(self.shape, other.shape)) {
            return TensorError.ShapeMismatch;
        }
        if (!self.device.eql(other.device)) {
            return TensorError.DeviceMismatch;
        }

        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad or other.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.sub(result.memory, self.memory, other.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const sub_backward = autograd.SubBackward.create(self.allocator, @constCast(self), @constCast(other)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &sub_backward.grad_fn;
        }

        return result;
    }

    /// Element-wise negation: -self
    pub fn neg(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.neg(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const neg_backward = autograd.NegBackward.create(self.allocator, @constCast(self)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &neg_backward.grad_fn;
        }

        return result;
    }

    /// ReLU activation: max(0, self)
    pub fn relu(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.relu(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const relu_backward = autograd.ReluBackward.create(self.allocator, @constCast(self)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &relu_backward.grad_fn;
        }

        return result;
    }

    /// Hyperbolic tangent: tanh(self)
    pub fn tanh(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.tanh(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const tanh_backward = autograd.TanhBackward.create(self.allocator, @constCast(self), &result) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &tanh_backward.grad_fn;
        }

        return result;
    }

    /// Sigmoid activation: 1 / (1 + exp(-self))
    pub fn sigmoid(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.sigmoid(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const sigmoid_backward = autograd.SigmoidBackward.create(self.allocator, @constCast(self), &result) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &sigmoid_backward.grad_fn;
        }

        return result;
    }

    /// Exponential: exp(self)
    pub fn exp(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.exp(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const exp_backward = autograd.ExpBackward.create(self.allocator, @constCast(self), &result) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &exp_backward.grad_fn;
        }

        return result;
    }

    /// Natural logarithm: log(self)
    pub fn log(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        result.requires_grad = self.requires_grad;

        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.log(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Create gradient function if needed
        if (result.requires_grad) {
            const log_backward = autograd.LogBackward.create(self.allocator, @constCast(self)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &log_backward.grad_fn;
        }

        return result;
    }

    /// Sine: sin(self)
    pub fn sin(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.sin(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };
        return result;
    }

    /// Cosine: cos(self)
    pub fn cos(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.cos(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };
        return result;
    }

    /// Square root: sqrt(self)
    pub fn sqrt(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.sqrt(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };
        return result;
    }

    /// Absolute value: abs(self)
    pub fn abs(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.abs(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };
        return result;
    }

    /// Sum all elements (returns scalar tensor)
    pub fn sum(self: *const Self) !Self {
        var result = try Self.initWithGrad(self.allocator, &.{}, self.device, self.requires_grad); // Scalar tensor
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        const sum_val = backend.sum(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Set the scalar value
        const result_data = try result.data();
        result_data[0] = sum_val;

        // Create gradient function if needed
        if (result.requires_grad) {
            const sum_backward = autograd.SumBackward.create(self.allocator, @constCast(self)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &sum_backward.grad_fn;
        }

        return result;
    }

    /// Mean of all elements (returns scalar tensor)
    pub fn mean(self: *const Self) !Self {
        var result = try Self.initWithGrad(self.allocator, &.{}, self.device, self.requires_grad); // Scalar tensor
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        const mean_val = backend.mean(result.memory, self.memory, self.num_elements) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        // Set the scalar value
        const result_data = try result.data();
        result_data[0] = mean_val;

        // Set up gradient function if needed
        if (result.requires_grad) {
            const mean_backward = autograd.MeanBackward.create(self.allocator, @constCast(self)) catch {
                result.deinit();
                return TensorError.OutOfMemory;
            };
            result.grad_fn = &mean_backward.grad_fn;
        }

        return result;
    }

    /// Multiply tensor by scalar
    pub fn mul_scalar(self: *const Self, scalar: f32) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const self_data = try self.constData();
        const result_data = try result.data();

        for (self_data, result_data) |val, *res| {
            res.* = val * scalar;
        }

        return result;
    }

    /// Clone tensor (deep copy)
    pub fn clone(self: *const Self) !Self {
        var result = try Self.init(self.allocator, self.shape, self.device);
        const self_data = try self.data();
        const result_data = try result.data();

        @memcpy(result_data, self_data);

        return result;
    }

    /// Matrix transpose (2D tensors only)
    pub fn transpose(self: *const Self) !Self {
        if (self.shape.len != 2) {
            return TensorError.ShapeMismatch;
        }

        const rows = self.shape[0];
        const cols = self.shape[1];

        // Result has transposed shape
        var result = try Self.init(self.allocator, &.{ cols, rows }, self.device);
        const backend = backend_mod.getBackend(self.device) catch {
            result.deinit();
            return TensorError.BackendError;
        };

        backend.transpose(result.memory, self.memory, rows, cols) catch {
            result.deinit();
            return TensorError.BackendError;
        };
        return result;
    }

    /// Backward pass - compute gradients
    pub fn backward(self: *Self) !void {
        // Initialize gradient if not set
        if (self.grad == null) {
            self.grad = try self.allocator.create(Tensor);
            self.grad.?.* = try ones(self.allocator, self.shape, self.device);
        }

        // Topological sort to determine execution order
        var topo_order = std.ArrayList(*Tensor).init(self.allocator);
        var visited = std.AutoHashMap(*Tensor, void).init(self.allocator);
        defer topo_order.deinit();
        defer visited.deinit();

        try buildTopo(self, &topo_order, &visited);

        // Execute backward pass in reverse topological order
        var i = topo_order.items.len;
        while (i > 0) {
            i -= 1;
            const tensor = topo_order.items[i];
            if (tensor.grad_fn != null and tensor.grad != null) {
                try tensor.grad_fn.?.backward(tensor.grad.?);
            }
        }
    }

    /// Zero out gradients
    pub fn zeroGrad(self: *Self) !void {
        if (self.grad) |grad| {
            try grad.fill(0.0);
        }
    }

    /// Get tensor shape
    pub fn getShape(self: *const Self) []const usize {
        return self.shape;
    }

    /// Get number of elements
    pub fn size(self: *const Self) usize {
        return self.num_elements;
    }

    /// Print tensor contents (CPU only)
    pub fn print(self: *const Self) !void {
        const stdout = std.io.getStdOut().writer();

        try stdout.print("Tensor(shape=[", .{});
        for (self.shape, 0..) |dim, i| {
            if (i > 0) try stdout.print(", ", .{});
            try stdout.print("{}", .{dim});
        }
        try stdout.print("], device={}, elements={})\n", .{ self.device, self.num_elements });

        if (self.device.isCpu()) {
            // Cast away const for data access - this is safe for reading
            var mutable_self = @constCast(self);
            const tensor_data = mutable_self.data() catch return;

            if (self.shape.len == 0) {
                // 0D tensor (scalar)
                try stdout.print("{d:.4}\n", .{tensor_data[0]});
            } else if (self.shape.len == 1) {
                // 1D tensor
                try stdout.print("[", .{});
                for (tensor_data, 0..) |val, i| {
                    if (i > 0) try stdout.print(", ", .{});
                    try stdout.print("{d:.4}", .{val});
                }
                try stdout.print("]\n", .{});
            } else if (self.shape.len == 2) {
                // 2D tensor
                const rows = self.shape[0];
                const cols = self.shape[1];
                try stdout.print("[\n", .{});
                for (0..rows) |i| {
                    try stdout.print("  [", .{});
                    for (0..cols) |j| {
                        if (j > 0) try stdout.print(", ", .{});
                        try stdout.print("{d:.4}", .{tensor_data[i * cols + j]});
                    }
                    try stdout.print("]\n", .{});
                }
                try stdout.print("]\n", .{});
            } else if (self.shape.len == 3) {
                // 3D tensor
                const depth = self.shape[0];
                const rows = self.shape[1];
                const cols = self.shape[2];
                try stdout.print("[\n", .{});
                for (0..depth) |d| {
                    try stdout.print("  [\n", .{});
                    for (0..rows) |i| {
                        try stdout.print("    [", .{});
                        for (0..cols) |j| {
                            if (j > 0) try stdout.print(", ", .{});
                            const idx = d * rows * cols + i * cols + j;
                            try stdout.print("{d:.4}", .{tensor_data[idx]});
                        }
                        try stdout.print("]\n", .{});
                    }
                    try stdout.print("  ]\n", .{});
                }
                try stdout.print("]\n", .{});
            } else {
                // Higher dimensional tensors - just print data array
                try stdout.print("Data: [", .{});
                for (tensor_data, 0..) |val, i| {
                    if (i > 0) try stdout.print(", ", .{});
                    try stdout.print("{d:.4}", .{val});
                    if (i >= 19) { // Limit output for very large tensors
                        try stdout.print(", ... ({} more elements)", .{tensor_data.len - 20});
                        break;
                    }
                }
                try stdout.print("]\n", .{});
            }
        }
    }
};

/// Create a zero-filled tensor
pub fn zeros(allocator: std.mem.Allocator, shape: []const usize, device: Device) !Tensor {
    return zerosWithGrad(allocator, shape, device, false);
}

/// Create a zero-filled tensor with gradient tracking control
pub fn zerosWithGrad(allocator: std.mem.Allocator, shape: []const usize, device: Device, requires_grad: bool) !Tensor {
    var tensor = try Tensor.initWithGrad(allocator, shape, device, requires_grad);
    try tensor.fill(0.0);
    return tensor;
}

/// Helper function for topological sort in backward pass
fn buildTopo(tensor: *Tensor, topo_order: *std.ArrayList(*Tensor), visited: *std.AutoHashMap(*Tensor, void)) !void {
    if (visited.contains(tensor)) return;

    try visited.put(tensor, {});

    if (tensor.grad_fn != null) {
        for (tensor.grad_fn.?.inputs.items) |input| {
            try buildTopo(input, topo_order, visited);
        }
    }

    try topo_order.append(tensor);
}

/// Create a one-filled tensor
pub fn ones(allocator: std.mem.Allocator, shape: []const usize, device: Device) !Tensor {
    return onesWithGrad(allocator, shape, device, false);
}

/// Create a one-filled tensor with gradient tracking control
pub fn onesWithGrad(allocator: std.mem.Allocator, shape: []const usize, device: Device, requires_grad: bool) !Tensor {
    var tensor = try Tensor.initWithGrad(allocator, shape, device, requires_grad);
    try tensor.fill(1.0);
    return tensor;
}

test "tensor creation and operations" {
    const allocator = std.testing.allocator;

    // Register CPU backend
    const cpu_backend = @import("../backend/cpu/cpu.zig").getCpuBackend();
    backend_mod.registerCpuBackend(cpu_backend);

    // Create tensors
    var a = try Tensor.init(allocator, &.{ 2, 2 }, Device.cpu());
    defer a.deinit();

    var b = try Tensor.init(allocator, &.{ 2, 2 }, Device.cpu());
    defer b.deinit();

    // Set data
    try a.setData(&.{ 1.0, 2.0, 3.0, 4.0 });
    try b.setData(&.{ 5.0, 6.0, 7.0, 8.0 });

    // Test addition
    var result = try a.add(&b);
    defer result.deinit();

    const result_data = try result.data();
    try std.testing.expectEqual(@as(f32, 6.0), result_data[0]);
    try std.testing.expectEqual(@as(f32, 8.0), result_data[1]);
    try std.testing.expectEqual(@as(f32, 10.0), result_data[2]);
    try std.testing.expectEqual(@as(f32, 12.0), result_data[3]);
}

test "tensor creation functions" {
    const allocator = std.testing.allocator;

    // Register CPU backend
    const cpu_backend = @import("../backend/cpu/cpu.zig").getCpuBackend();
    backend_mod.registerCpuBackend(cpu_backend);

    // Test zeros
    var zero_tensor = try zeros(allocator, &.{ 2, 3 }, Device.cpu());
    defer zero_tensor.deinit();

    const zero_data = try zero_tensor.data();
    for (zero_data) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }

    // Test ones
    var one_tensor = try ones(allocator, &.{ 2, 3 }, Device.cpu());
    defer one_tensor.deinit();

    const one_data = try one_tensor.data();
    for (one_data) |val| {
        try std.testing.expectEqual(@as(f32, 1.0), val);
    }
}
