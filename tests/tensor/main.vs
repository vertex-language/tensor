// tensor on every device: tensors made from bytes, read as floats, shapes
// checked, and Add.
package main

import "gpu"
import "tensor"

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

for d in [gpu.CPU(), gpu.Default()] {
    let host = try await d.Upload([float32(1), 2, 3, 4, 5, 6])
    let bytes = UnsafePointer<uint8>(UnsafeMutableRawPointer(host._elements))
    let t = try await tensor.Tensor.FromBytes(bytes, shape: [2, 3], dtype: .F32, on: d)
    check(t.Count == 6 && t.RowBytes == 12 && t.Shape == [2, 3], "\(d.Name): a [2, 3] f32 tensor from bytes")
    check(try await t.Floats().Download() == [1, 2, 3, 4, 5, 6], "\(d.Name): its floats")
    let z = try await tensor.Tensor.Zeros([6], on: d)
    try await tensor.Add(try t.Floats(), try t.Floats(), into: try z.Floats())
    check(try await z.Floats().Download() == [2, 4, 6, 8, 10, 12], "\(d.Name): Add")
    let c = try await tensor.ConcatRows([t, t])
    check(c.Shape == [4, 3] && (try await c.Floats().Download()) == [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], "\(d.Name): ConcatRows")
    let q = try await tensor.Tensor.Zeros([18], on: d)
    check(tensor.DType.Q4_0.Bytes(64) == 36 && tensor.DType.Q8_0.Bytes(32) == 34, "\(d.Name): block sizes")
    do {
        _ = try tensor.Tensor(shape: [4, 33], dtype: .Q4_0, storage: q.Storage)
        check(false, "\(d.Name): a row of 33 q4_0 refused")
    } catch {
        check(true, "\(d.Name): a row of 33 q4_0 refused")
    }
    do {
        _ = try t.Floats()
        _ = try tensor.Tensor(shape: [2, 32], dtype: .Q8_0, storage: q.Storage).Floats()
        check(false, "\(d.Name): q8_0 read as floats refused")
    } catch {
        check(true, "\(d.Name): q8_0 read as floats refused")
    }
}
print(failures == 0 ? "all passed" : "\(failures) failed")
