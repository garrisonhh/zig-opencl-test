const std = @import("std");
const stdout = std.io.getStdOut().writer();
const stderr = std.io.getStdOut().writer();
const cl = @import("cl.zig");

/// sums two lists of floats
const SOURCE =
    \\__kernel void add(
    \\    __global float *a,
    \\    __global float *b,
    \\    __global float *output,
    \\    unsigned int len
    \\) {
    \\    int gid = get_global_id(0);
    \\    if (gid < len) {
    \\        output[gid] = a[gid] + b[gid];
    \\    }
    \\}
;

fn createBuffer(
    context: cl.cl_context,
    flags: cl.cl_mem_flags,
    comptime T: type,
    len: usize,
) cl.Error!cl.cl_mem {
    var err: cl.cl_int = undefined;
    const size = @sizeOf(T) * len;
    const mem = cl.clCreateBuffer(context, flags, size, null, &err);
    try cl.wrap(err);

    return mem;
}

fn setKernelArgs(kernel: cl.cl_kernel, args: anytype) cl.Error!void {
    inline for (args) |arg, i| {
        try cl.wrap(cl.clSetKernelArg(
            kernel,
            i,
            @sizeOf(@TypeOf(arg.*)),
            @ptrCast(*const anyopaque, arg),
        ));
    }
}

var rng = std.rand.DefaultPrng.init(0);

pub fn main() !void {
    // generate input data
    const COUNT: cl.cl_uint = 16;

    var in_a: [COUNT]f32 = undefined;
    var in_b: [COUNT]f32 = undefined;

    var i: usize = 0;
    while (i < COUNT) : (i += 1) {
        in_a[i] = rng.random().float(f32);
        in_b[i] = rng.random().float(f32);
    }

    var err: cl.cl_int = undefined;

    // connect to compute device
    var device: cl.cl_device_id = undefined;
    try cl.wrap(cl.clGetDeviceIDs(null, cl.CL_TRUE, 1, &device, null));

    // create compute context
    const context = cl.clCreateContext(null, 1, &device, null, null, &err);
    try cl.wrap(err);
    defer _ = cl.clReleaseContext(context);

    // create command queue
    const commands = cl.clCreateCommandQueue(context, device, 0, &err);
    try cl.wrap(err);
    defer _ = cl.clReleaseCommandQueue(commands);

    // create and build program
    var sources = [_][*]const u8{SOURCE};
    const program = cl.clCreateProgramWithSource(
        context,
        sources.len,
        @ptrCast([*c][*c]const u8, &sources),
        null,
        &err,
    );

    try cl.wrap(err);
    defer _ = cl.clReleaseProgram(program);

    err = cl.clBuildProgram(program, 1, &device, null, null, null);
    cl.wrap(err) catch |e| {
        var buf: [1024]u8 = undefined;
        var len: usize = undefined;

        try cl.wrap(cl.clGetProgramBuildInfo(
            program,
            device,
            cl.CL_PROGRAM_BUILD_LOG,
            @sizeOf(@TypeOf(buf)),
            &buf,
            &len,
        ));

        const msg = buf[0..len];
        try stderr.print("error building program:\n{s}\n", .{msg});

        return e;
    };

    // create compute kernel for program
    const kernel = cl.clCreateKernel(program, "add", &err);
    try cl.wrap(err);
    defer _ = cl.clReleaseKernel(kernel);

    // create buffers
    var buf_a = try createBuffer(context, cl.CL_MEM_READ_ONLY, f32, COUNT);
    defer _ = cl.clReleaseMemObject(buf_a);
    var buf_b = try createBuffer(context, cl.CL_MEM_READ_ONLY, f32, COUNT);
    defer _ = cl.clReleaseMemObject(buf_b);
    var buf_out = try createBuffer(context, cl.CL_MEM_WRITE_ONLY, f32, COUNT);
    defer _ = cl.clReleaseMemObject(buf_out);

    // write input to buffers
    try cl.wrap(cl.clEnqueueWriteBuffer(
        commands,
        buf_a,
        cl.CL_TRUE,
        0,
        @sizeOf(@TypeOf(in_a)),
        &in_a,
        0,
        null,
        null,
    ));
    try cl.wrap(cl.clEnqueueWriteBuffer(
        commands,
        buf_b,
        cl.CL_TRUE,
        0,
        @sizeOf(@TypeOf(in_b)),
        &in_b,
        0,
        null,
        null,
    ));

    // set the kernel argumentis
    try setKernelArgs(kernel, .{ &buf_a, &buf_b, &buf_out, &COUNT });

    // get max kernel work group size
    var local: usize = undefined;
    try cl.wrap(cl.clGetKernelWorkGroupInfo(
        kernel,
        device,
        cl.CL_KERNEL_WORK_GROUP_SIZE,
        @sizeOf(@TypeOf(local)),
        &local,
        null,
    ));

    // execute the kernel over the range of input ids
    var global: usize = @max(local, COUNT);
    try cl.wrap(cl.clEnqueueNDRangeKernel(
        commands,
        kernel,
        1,
        null,
        &global,
        &local,
        0,
        null,
        null,
    ));

    // wait for commands to finish
    try cl.wrap(cl.clFinish(commands));

    // read the results back
    var out: [COUNT]f32 = undefined;
    try cl.wrap(cl.clEnqueueReadBuffer(
        commands,
        buf_out,
        cl.CL_TRUE,
        0,
        @sizeOf(@TypeOf(out)),
        &out,
        0,
        null,
        null,
    ));

    // display output
    try stdout.print(
        \\[computed results]
        \\a:   {d:0.4}
        \\b:   {d:0.4}
        \\out: {d:0.4}
        \\
    ,
        .{in_a, in_b, out},
    );
}
