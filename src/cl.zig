const std = @import("std");

const cl = @cImport({
    @cDefine("CL_TARGET_OPENCL_VERSION", "300");
    @cInclude("CL/cl.h");
});

pub usingnamespace cl;

/// derived from CL/cl.h
pub const Error = error {
    DeviceNotFound,
    DeviceNotAvailable,
    CompilerNotAvailable,
    MemObjectAllocationFailure,
    OutOfResources,
    OutOfHostMemory,
    ProfilingInfoNotAvailable,
    MemCopyOverlap,
    ImageFormatMismatch,
    ImageFormatNotSupported,
    BuildProgramFailure,
    MapFailure,
    MisalignedSubBufferOffset,
    ExecStatusErrorForEventsInWaitList,
    CompileProgramFailure,
    LinkerNotAvailable,
    LinkProgramFailure,
    DevicePartitionFailed,
    KernelArgInfoNotAvailable,
    InvalidValue,
    InvalidDeviceType,
    InvalidPlatform,
    InvalidDevice,
    InvalidContext,
    InvalidQueueProperties,
    InvalidCommandQueue,
    InvalidHostPtr,
    InvalidMemObject,
    InvalidImageFormatDescriptor,
    InvalidImageSize,
    InvalidSampler,
    InvalidBinary,
    InvalidBuildOptions,
    InvalidProgram,
    InvalidProgramExecutable,
    InvalidKernelName,
    InvalidKernelDefinition,
    InvalidKernel,
    InvalidArgIndex,
    InvalidArgValue,
    InvalidArgSize,
    InvalidKernelArgs,
    InvalidWorkDimension,
    InvalidWorkGroupSize,
    InvalidWorkItemSize,
    InvalidGlobalOffset,
    InvalidEventWaitList,
    InvalidEvent,
    InvalidOperation,
    InvalidGlObject,
    InvalidBufferSize,
    InvalidMipLevel,
    InvalidGlobalWorkSize,
    InvalidProperty,
    InvalidImageDescriptor,
    InvalidCompilerOptions,
    InvalidLinkerOptions,
    InvalidDevicePartitionCount,
    InvalidPipeSize,
    InvalidDeviceQueue,
    InvalidSpecId,
    MaxSizeRestrictionExceeded,
};

/// derived from CL/cl.h
pub fn decodeError(code: cl.cl_int) Error {
    return switch (code) {
        -1 => Error.DeviceNotFound,
        -2 => Error.DeviceNotAvailable,
        -3 => Error.CompilerNotAvailable,
        -4 => Error.MemObjectAllocationFailure,
        -5 => Error.OutOfResources,
        -7 => Error.ProfilingInfoNotAvailable,
        -8 => Error.MemCopyOverlap,
        -10 => Error.ImageFormatNotSupported,
        -11 => Error.BuildProgramFailure,
        -12 => Error.MapFailure,
        -13 => Error.MisalignedSubBufferOffset,
        -14 => Error.ExecStatusErrorForEventsInWaitList,
        -15 => Error.CompileProgramFailure,
        -16 => Error.LinkerNotAvailable,
        -17 => Error.LinkProgramFailure,
        -18 => Error.DevicePartitionFailed,
        -19 => Error.KernelArgInfoNotAvailable,
        -30 => Error.InvalidValue,
        -31 => Error.InvalidDeviceType,
        -32 => Error.InvalidPlatform,
        -33 => Error.InvalidDevice,
        -34 => Error.InvalidContext,
        -35 => Error.InvalidQueueProperties,
        -36 => Error.InvalidCommandQueue,
        -37 => Error.InvalidHostPtr,
        -38 => Error.InvalidMemObject,
        -39 => Error.InvalidImageFormatDescriptor,
        -40 => Error.InvalidImageSize,
        -41 => Error.InvalidSampler,
        -42 => Error.InvalidBinary,
        -43 => Error.InvalidBuildOptions,
        -44 => Error.InvalidProgram,
        -45 => Error.InvalidProgramExecutable,
        -46 => Error.InvalidKernelName,
        -47 => Error.InvalidKernelDefinition,
        -48 => Error.InvalidKernel,
        -49 => Error.InvalidArgIndex,
        -50 => Error.InvalidArgValue,
        -51 => Error.InvalidArgSize,
        -52 => Error.InvalidKernelArgs,
        -53 => Error.InvalidWorkDimension,
        -54 => Error.InvalidWorkGroupSize,
        -55 => Error.InvalidWorkItemSize,
        -56 => Error.InvalidGlobalOffset,
        -57 => Error.InvalidEventWaitList,
        -58 => Error.InvalidEvent,
        -59 => Error.InvalidOperation,
        -60 => Error.InvalidGlObject,
        -61 => Error.InvalidBufferSize,
        -62 => Error.InvalidMipLevel,
        -63 => Error.InvalidGlobalWorkSize,
        -64 => Error.InvalidProperty,
        -65 => Error.InvalidImageDescriptor,
        -66 => Error.InvalidCompilerOptions,
        -67 => Error.InvalidLinkerOptions,
        -68 => Error.InvalidDevicePartitionCount,
        -69 => Error.InvalidPipeSize,
        -70 => Error.InvalidDeviceQueue,
        -71 => Error.InvalidSpecId,
        -72 => Error.MaxSizeRestrictionExceeded,
        else => std.debug.panic("error: unknown cl error code {d}", .{code}),
    };
}

/// convenience function for wrapping stuff
/// ```zig
/// try wrap(cl_call(xyz));
/// ```
pub fn wrap(err: cl.cl_int) Error!void {
    return if (err == 0) {} else decodeError(err);
}
