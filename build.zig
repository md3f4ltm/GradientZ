const std = @import("std");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    // This creates a "module", which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Every executable or library we compile will be based on one or more modules.
    const lib_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // We will also create a module for our other entry point, 'main.zig'.
    const exe_mod = b.createModule(.{
        // `root_source_file` is the Zig "entry point" of the module. If a module
        // only contains e.g. external object files, you can make this `null`.
        // In this case the main source file is merely a path, however, in more
        // complicated build scripts, this could be a generated file.
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create modules for all examples
    const pytorch_example_mod = b.createModule(.{
        .root_source_file = b.path("examples/pytorch_like.zig"),
        .target = target,
        .optimize = optimize,
    });

    const scalars_0d_mod = b.createModule(.{
        .root_source_file = b.path("examples/scalars_0d.zig"),
        .target = target,
        .optimize = optimize,
    });

    const vectors_1d_mod = b.createModule(.{
        .root_source_file = b.path("examples/vectors_1d.zig"),
        .target = target,
        .optimize = optimize,
    });

    const matrices_2d_mod = b.createModule(.{
        .root_source_file = b.path("examples/matrices_2d.zig"),
        .target = target,
        .optimize = optimize,
    });

    const tensors_3d_mod = b.createModule(.{
        .root_source_file = b.path("examples/tensors_3d.zig"),
        .target = target,
        .optimize = optimize,
    });

    const autograd_mod = b.createModule(.{
        .root_source_file = b.path("examples/autograd.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mlp_mod = b.createModule(.{
        .root_source_file = b.path("examples/mlp.zig"),
        .target = target,
        .optimize = optimize,
    });

    const devices_mod = b.createModule(.{
        .root_source_file = b.path("examples/devices.zig"),
        .target = target,
        .optimize = optimize,
    });

    const training_mod = b.createModule(.{
        .root_source_file = b.path("examples/training.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Modules can depend on one another using the `std.Build.Module.addImport` function.
    // This is what allows Zig source code to use `@import("foo")` where 'foo' is not a
    // file path. In this case, we set up `exe_mod` to import `lib_mod`.
    exe_mod.addImport("GradientZ_lib", lib_mod);
    pytorch_example_mod.addImport("GradientZ_lib", lib_mod);
    scalars_0d_mod.addImport("GradientZ_lib", lib_mod);
    vectors_1d_mod.addImport("GradientZ_lib", lib_mod);
    matrices_2d_mod.addImport("GradientZ_lib", lib_mod);
    tensors_3d_mod.addImport("GradientZ_lib", lib_mod);
    autograd_mod.addImport("GradientZ_lib", lib_mod);
    mlp_mod.addImport("GradientZ_lib", lib_mod);
    devices_mod.addImport("GradientZ_lib", lib_mod);
    training_mod.addImport("GradientZ_lib", lib_mod);

    // Now, we will create a static library based on the module we created above.
    // This creates a `std.Build.Step.Compile`, which is the build step responsible
    // for actually invoking the compiler.
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "GradientZ",
        .root_module = lib_mod,
    });

    // This declares intent for the library to be installed into the standard
    // location when the user invokes the "install" step (the default step when
    // running `zig build`).
    b.installArtifact(lib);

    // This creates another `std.Build.Step.Compile`, but this one builds an executable
    // rather than a static library.
    const exe = b.addExecutable(.{
        .name = "GradientZ",
        .root_module = exe_mod,
    });

    // Create all example executables
    const pytorch_example = b.addExecutable(.{
        .name = "pytorch_example",
        .root_module = pytorch_example_mod,
    });

    const scalars_0d_example = b.addExecutable(.{
        .name = "scalars_0d_example",
        .root_module = scalars_0d_mod,
    });

    const vectors_1d_example = b.addExecutable(.{
        .name = "vectors_1d_example",
        .root_module = vectors_1d_mod,
    });

    const matrices_2d_example = b.addExecutable(.{
        .name = "matrices_2d_example",
        .root_module = matrices_2d_mod,
    });

    const tensors_3d_example = b.addExecutable(.{
        .name = "tensors_3d_example",
        .root_module = tensors_3d_mod,
    });

    const autograd_example = b.addExecutable(.{
        .name = "autograd_example",
        .root_module = autograd_mod,
    });

    const mlp_example = b.addExecutable(.{
        .name = "mlp_example",
        .root_module = mlp_mod,
    });

    const devices_example = b.addExecutable(.{
        .name = "devices_example",
        .root_module = devices_mod,
    });

    const training_example = b.addExecutable(.{
        .name = "training_example",
        .root_module = training_mod,
    });

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);
    b.installArtifact(pytorch_example);
    b.installArtifact(scalars_0d_example);
    b.installArtifact(vectors_1d_example);
    b.installArtifact(matrices_2d_example);
    b.installArtifact(tensors_3d_example);
    b.installArtifact(autograd_example);
    b.installArtifact(mlp_example);
    b.installArtifact(devices_example);
    b.installArtifact(training_example);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Create run steps for all examples
    const run_pytorch_cmd = b.addRunArtifact(pytorch_example);
    run_pytorch_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_pytorch_cmd.addArgs(args);
    }

    const run_scalars_0d_cmd = b.addRunArtifact(scalars_0d_example);
    run_scalars_0d_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_scalars_0d_cmd.addArgs(args);
    }

    const run_vectors_1d_cmd = b.addRunArtifact(vectors_1d_example);
    run_vectors_1d_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_vectors_1d_cmd.addArgs(args);
    }

    const run_matrices_2d_cmd = b.addRunArtifact(matrices_2d_example);
    run_matrices_2d_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_matrices_2d_cmd.addArgs(args);
    }

    const run_tensors_3d_cmd = b.addRunArtifact(tensors_3d_example);
    run_tensors_3d_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_tensors_3d_cmd.addArgs(args);
    }

    const run_autograd_cmd = b.addRunArtifact(autograd_example);
    run_autograd_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_autograd_cmd.addArgs(args);
    }

    const run_mlp_cmd = b.addRunArtifact(mlp_example);
    run_mlp_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_mlp_cmd.addArgs(args);
    }

    const run_devices_cmd = b.addRunArtifact(devices_example);
    run_devices_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_devices_cmd.addArgs(args);
    }

    const run_training_cmd = b.addRunArtifact(training_example);
    run_training_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_training_cmd.addArgs(args);
    }

    // Create build steps for examples
    const run_pytorch_step = b.step("example", "Run the PyTorch-like example");
    run_pytorch_step.dependOn(&run_pytorch_cmd.step);

    const run_scalars_0d_step = b.step("example-scalars", "Run the 0D tensors (scalars) example");
    run_scalars_0d_step.dependOn(&run_scalars_0d_cmd.step);

    const run_vectors_1d_step = b.step("example-vectors", "Run the 1D tensors (vectors) example");
    run_vectors_1d_step.dependOn(&run_vectors_1d_cmd.step);

    const run_matrices_2d_step = b.step("example-matrices", "Run the 2D tensors (matrices) example");
    run_matrices_2d_step.dependOn(&run_matrices_2d_cmd.step);

    const run_tensors_3d_step = b.step("example-3d", "Run the 3D tensors example");
    run_tensors_3d_step.dependOn(&run_tensors_3d_cmd.step);

    const run_autograd_step = b.step("example-autograd", "Run the autograd (automatic differentiation) example");
    run_autograd_step.dependOn(&run_autograd_cmd.step);

    const run_mlp_step = b.step("example-mlp", "Run the MLP (Multi-Layer Perceptron) example");
    run_mlp_step.dependOn(&run_mlp_cmd.step);

    const run_devices_step = b.step("example-devices", "Run the device functionality example");
    run_devices_step.dependOn(&run_devices_cmd.step);

    const run_training_step = b.step("example-training", "Run the training (gradient descent) example");
    run_training_step.dependOn(&run_training_cmd.step);

    const run_all_examples_step = b.step("examples", "Run all examples");
    run_all_examples_step.dependOn(&run_pytorch_cmd.step);
    run_all_examples_step.dependOn(&run_scalars_0d_cmd.step);
    run_all_examples_step.dependOn(&run_vectors_1d_cmd.step);
    run_all_examples_step.dependOn(&run_matrices_2d_cmd.step);
    run_all_examples_step.dependOn(&run_tensors_3d_cmd.step);
    run_all_examples_step.dependOn(&run_autograd_cmd.step);
    run_all_examples_step.dependOn(&run_mlp_cmd.step);
    run_all_examples_step.dependOn(&run_devices_cmd.step);
    run_all_examples_step.dependOn(&run_training_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
