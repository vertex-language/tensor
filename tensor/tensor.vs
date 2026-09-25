// Package tensor is the n-d array: a shape, an element type, and bytes on
// a device. Weights are Tensors whatever their format -- float32 or
// block-quantized -- and the layers in nn dispatch on DType.
//
// This is the first cut, grown from what running a llama needs: a weight
// tensor made from a file's bytes, read as floats or as quantized blocks,
// and elementwise addition. Views, strides and the eager op set of
// proposed_ai_packages.md §6.1 come as the models ahead need them.
package tensor

import "gpu"
import "gpu/dtype"

/// DType is how a tensor's elements are stored.
public enum DType {
    case F32
    case F16
    case BF16
    case Q4_0
    case Q8_0

    /// Name is ggml's name for it: "f32", "q4_0".
    public var Name: string {
        switch self {
        case .F32: return "f32"
        case .F16: return "f16"
        case .BF16: return "bf16"
        case .Q4_0: return "q4_0"
        case .Q8_0: return "q8_0"
        }
    }

    /// BlockSize is how many elements share a block: 1 for a scalar type.
    public var BlockSize: int {
        switch self {
        case .Q4_0, .Q8_0: return 32
        default: return 1
        }
    }

    /// BlockBytes is how many bytes a block takes.
    public var BlockBytes: int {
        switch self {
        case .F32: return 4
        case .F16, .BF16: return 2
        case .Q4_0: return dtype.Q4_0.Bytes()
        case .Q8_0: return dtype.Q8_0.Bytes()
        }
    }

    /// Bytes is how many bytes n elements take; n is a multiple of
    /// BlockSize.
    public func Bytes(_ n: int) -> int {
        return n / BlockSize * BlockBytes
    }
}

/// ShapeError is a tensor made or used with a shape that does not fit.
public enum ShapeError: Error {
    case mismatch(string)

    public var Message: string {
        switch self {
        case .mismatch(let why): return "tensor: " + why
        }
    }
}

/// Tensor is an n-d array on a device, row-major: Shape is outermost
/// first, as PyTorch writes it, so a [out, in] weight is out rows of in
/// elements. It is a handle; its bytes go with the last reference.
public final class Tensor {
    public let Shape: [int]
    public let DType: DType
    /// Storage is the tensor's bytes, as the format lays them out.
    public let Storage: gpu.Buffer<uint8>

    public init(shape: [int], dtype: DType, storage: gpu.Buffer<uint8>) throws {
        self.Shape = shape
        self.DType = dtype
        self.Storage = storage
        var n = 1
        for d in shape { n *= d }
        if shape.count > 0 && shape[shape.count - 1] % dtype.BlockSize != 0 {
            throw ShapeError.mismatch("a row of \(shape[shape.count - 1]) is not whole \(dtype.Name) blocks of \(dtype.BlockSize)")
        }
        if dtype.Bytes(n) != storage.count {
            throw ShapeError.mismatch("\(shape) of \(dtype.Name) is \(dtype.Bytes(n)) bytes, not \(storage.count)")
        }
    }

    /// Count is how many elements the tensor holds.
    public var Count: int {
        var n = 1
        for d in Shape { n *= d }
        return n
    }

    /// Device is where the tensor's bytes are.
    public var Device: gpu.Device { return Storage.Device }

    /// RowBytes is the bytes of one innermost row.
    public var RowBytes: int {
        return DType.Bytes(Shape.count > 0 ? Shape[Shape.count - 1] : 1)
    }

    /// Floats is a float32 tensor's elements, the same memory.
    public func Floats() throws -> gpu.Buffer<float32> {
        if DType != .F32 {
            throw ShapeError.mismatch("a \(DType.Name) tensor read as float32")
        }
        return Storage.View(as: float32.self)
    }

    /// FromBytes is a tensor of shape and dtype on device d, its bytes
    /// copied from host memory laid out as the format lays them: a mapped
    /// file's tensor.
    public static func FromBytes(_ bytes: UnsafePointer<uint8>, shape: [int], dtype: DType, on d: gpu.Device) async throws -> Tensor {
        var n = 1
        for s in shape { n *= s }
        let storage = try await d.Upload(from: bytes, count: dtype.Bytes(n))
        return try Tensor(shape: shape, dtype: dtype, storage: storage)
    }

    /// Zeros is a float32 tensor of shape, every element 0.
    public static func Zeros(_ shape: [int], on d: gpu.Device) async throws -> Tensor {
        var n = 1
        for s in shape { n *= s }
        let storage = try d.CreateBuffer(of: uint8.self, count: 4 * n)
        try await storage.Fill(0)
        return try Tensor(shape: shape, dtype: .F32, storage: storage)
    }
}

func _add(_ a: gpu.Span<float32>, _ b: gpu.Span<float32>, _ y: gpu.MutableSpan<float32>) kernel {
    let i = gpu.Index.x
    if i < y.count {
        y[i] = a[i] + b[i]
    }
}

/// Add writes a + b into y, element by element; y may be a or b. The
/// residual connection of every transformer block.
public func Add(_ a: gpu.Buffer<float32>, _ b: gpu.Buffer<float32>, into y: gpu.Buffer<float32>) async throws {
    if a.count != y.count || b.count != y.count {
        throw ShapeError.mismatch("Add of \(a.count) and \(b.count) into \(y.count)")
    }
    if y.count == 0 { return }
    try await _add.Launch(a, b, y, over: y.count)
}
