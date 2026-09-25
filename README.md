# tensor

The n-d array, on a device (`proposed_ai_packages.md` §6.1). It grows from
what running models needs. Today that is `model/llama`'s decode.

| Package | Built | Tested |
| --- | --- | --- |
| `tensor` | `Tensor`: a shape (outermost first, row-major), a `DType` (`F32`, `F16`, `BF16`, and the block formats `Q4_0` and `Q8_0` as ggml lays them out), and bytes on a `gpu.Device`. It is made from host bytes (`FromBytes`, a mapped weight file's tensor) or as `Zeros`, and read as floats in place (`Floats`). Shapes are checked against the format's blocks. `ConcatRows` fuses weights of one input (QKV, gate and up). `Add` adds elementwise | `test-tensor` on the CPU device and Metal |

Views, strides, the eager op set, `tensor/quant`, `tensor/autodiff` and
`tensor/graph` come as the models need them.

```console
$ vsc run test-tensor
```
