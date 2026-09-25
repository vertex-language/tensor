# tensor

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![tensor: multi-device](https://img.shields.io/badge/tensor-multi--device-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/tensor)

Multidimensional array operations and device buffers: shapes, dtypes, memory layouts, and elementwise operations across compute devices.

---

## Quick Start

Run any entry point with:

```bash
vsc run main.vs
```

### Running Tests

```bash
vsc run test-tensor
```

---

## Packages

| Package | Built | Tested |
| --- | --- | --- |
| `tensor` | `Tensor`: a shape (outermost first, row-major), a `DType` (`F32`, `F16`, `BF16`, and the block formats `Q4_0` and `Q8_0` as ggml lays them out), and bytes on a `gpu.Device`. It is made from host bytes (`FromBytes`, a mapped weight file's tensor) or as `Zeros`, and read as floats in place (`Floats`). Shapes are checked against the format's blocks. `ConcatRows` fuses weights of one input (QKV, gate and up). `Add` adds elementwise | `test-tensor` on the CPU device and Metal |

---

## License

[MIT](LICENSE)
