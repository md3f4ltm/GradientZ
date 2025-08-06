const std = @import("std");

pub const ShapeError = error{
    InvalidShape,
    EmptyShape,
    ShapeMismatch,
};

// Calculate total number of elements in a shape

pub fn calcElements(shape: []const usize) ShapeError!usize {
    // 0D tensor (scalar) has 1 element
    if (shape.len == 0) {
        return 1;
    }

    var total: usize = 1;
    for (shape) |dim| {
        if (dim == 0) {
            return ShapeError.InvalidShape;
        }
        total *= dim;
    }

    return total;
}

// Check if two shapes are equal
pub fn shapesEqual(a: []const usize, b: []const usize) bool {
    return std.mem.eql(usize, a, b);
}

// Copy shape slice

pub fn copyShape(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
    return try allocator.dupe(usize, shape);
}

test "shape utilities" {
    // Test element calculation
    try std.testing.expectEqual(@as(usize, 1), try calcElements(&.{})); // 0D tensor
    try std.testing.expectEqual(@as(usize, 6), try calcElements(&.{ 2, 3 }));
    try std.testing.expectEqual(@as(usize, 24), try calcElements(&.{ 2, 3, 4 }));

    // Test shape equality
    try std.testing.expect(shapesEqual(&.{ 2, 3 }, &.{ 2, 3 }));
    try std.testing.expect(!shapesEqual(&.{ 2, 3 }, &.{ 3, 2 }));
}

/// Broadcasting result for two shapes
pub const BroadcastResult = struct {
    output_shape: []usize,
    can_broadcast: bool,

    pub fn deinit(self: *BroadcastResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output_shape);
    }
};

/// Check if two shapes can be broadcast together and compute result shape
pub fn computeBroadcast(allocator: std.mem.Allocator, shape_a: []const usize, shape_b: []const usize) !BroadcastResult {
    // Align shapes from the right (trailing dimensions)
    const max_dims = @max(shape_a.len, shape_b.len);
    var output_shape = try allocator.alloc(usize, max_dims);

    for (0..max_dims) |i| {
        const dim_idx = max_dims - 1 - i;

        // Get dimension sizes (1 if dimension doesn't exist)
        const a_size = if (i < shape_a.len) shape_a[shape_a.len - 1 - i] else 1;
        const b_size = if (i < shape_b.len) shape_b[shape_b.len - 1 - i] else 1;

        // Broadcasting rules
        if (a_size == b_size) {
            output_shape[dim_idx] = a_size;
        } else if (a_size == 1) {
            output_shape[dim_idx] = b_size;
        } else if (b_size == 1) {
            output_shape[dim_idx] = a_size;
        } else {
            allocator.free(output_shape);
            return BroadcastResult{ .output_shape = &.{}, .can_broadcast = false };
        }
    }

    return BroadcastResult{ .output_shape = output_shape, .can_broadcast = true };
}

/// canBroadcast function
pub fn canBroadcast(shape_a: []const usize, shape_b: []const usize) bool {
    const max_dims = @max(shape_a.len, shape_b.len);

    for (0..max_dims) |i| {
        const a_size = if (i < shape_a.len) shape_a[shape_a.len - 1 - i] else 1;
        const b_size = if (i < shape_b.len) shape_b[shape_b.len - 1 - i] else 1;

        if (a_size != b_size and a_size != 1 and b_size != 1) {
            return false;
        }
    }
    return true;
}
