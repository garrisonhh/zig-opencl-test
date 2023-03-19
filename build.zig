const std = @import("std");

fn addOpenCL(step: *std.build.LibExeObjStep) void {
    step.addIncludePath("lib");
    step.linkSystemLibrary("OpenCL");
    step.linkLibC();
}

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("tic", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    addOpenCL(exe);
    exe.install();

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_tests = b.addTest("src/main.zig");
    exe_tests.setTarget(target);
    exe_tests.setBuildMode(mode);
    addOpenCL(exe_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&exe_tests.step);
}
